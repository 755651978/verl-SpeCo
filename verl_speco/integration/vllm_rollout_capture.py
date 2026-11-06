# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in V1 rollout feature capture with bounded async export (vLLM 0.22.0).

The sampler hook runs before the proposer can reuse target output buffers.
Only intersecting window rows are copied to CPU and filtered before export.
Owned GPU snapshots protect graph outputs while D2H/export overlap execution.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

CAPTURE_KEY = "speco_rollout_capture"
CAPTURE_FLAG = "collect_hidden_states_from_vllm"
logger = logging.getLogger(__name__)
_EXPORT_QUEUE: Any = None


def export_queue():
    global _EXPORT_QUEUE
    if _EXPORT_QUEUE is None:
        from verl_speco.producer.feature_export import FeatureExportQueue

        _EXPORT_QUEUE = FeatureExportQueue()
    return _EXPORT_QUEUE


def enabled(config: Any) -> bool:
    return bool((config.get("training") or {}).get(CAPTURE_FLAG, False))


def validate_config(drafter: Any, rollout: Any, engine: Any) -> None:
    if not enabled(drafter):
        return
    training = drafter.get("training") or {}
    spec = engine.get("speculative_config") or {}
    if spec.get("method") == "extract_hidden_states":
        raise ValueError(
            "Direct rollout capture cannot use method=extract_hidden_states (#62411)"
        )
    connector = engine.get("kv_transfer_config") or {}
    if connector.get("kv_connector") == "ExampleHiddenStatesConnector":
        raise ValueError(
            "Direct rollout capture cannot use ExampleHiddenStatesConnector"
        )
    if not drafter.get("enable") or not drafter.get("enable_drafter_training"):
        raise ValueError("vLLM rollout capture requires drafter training enabled")
    if any(
        training.get(k)
        for k in (
            "collect_hidden_states_from_sgl",
            "collect_hidden_states_from_old_logprob",
            "use_logits",
        )
    ):
        raise ValueError(
            "vLLM rollout capture requires exclusive hidden collection and use_logits=false"
        )
    if str(drafter.get("speculative_algorithm", "")).upper() not in {
        "EAGLE3",
        "DFLASH",
    }:
        raise ValueError(
            "vLLM rollout capture initially supports EAGLE3 and DFLASH only"
        )
    for key in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "data_parallel_size",
    ):
        if int(rollout.get(key, 1) or 1) != 1:
            raise ValueError(f"vLLM rollout capture requires {key}=1")
    if engine.get("async_scheduling") is not False:
        raise ValueError(
            "vLLM rollout capture requires engine_kwargs.vllm.async_scheduling=false"
        )
    if rollout.get("enable_prefix_caching", True) is not False:
        raise ValueError(
            "vLLM rollout capture requires rollout.enable_prefix_caching=false"
        )
    if str(training.get("hidden_state_window_mode", "front")) not in {
        "front",
        "random",
    }:
        raise ValueError("vLLM rollout capture supports front/random windows")
    if int(training.get("hidden_state_window_tokens_per_sample", 512)) <= 0:
        raise ValueError("vLLM capture window must be positive and bounded")
    if (
        not 1
        <= int(training.get("hidden_state_window_min_rows", 1))
        <= int(training.get("hidden_state_window_tokens_per_sample", 512))
    ):
        raise ValueError(
            "vLLM capture min_rows must be positive and fit within the window"
        )
    if training.get("hidden_state_max_tokens_per_sample") is not None:
        logger.warning(
            "vLLM capture ignores hidden_state_max_tokens_per_sample; use hidden_state_window_tokens_per_sample"
        )
    if training.get("preserve_prompt_hidden_states"):
        raise ValueError(
            "vLLM rollout capture currently collects response prediction rows only"
        )
    if os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0") == "1":
        raise ValueError("vLLM rollout capture requires the V1 model runner")


def valid_row_count(scheduled: int, drafts: int, sampled: list[int], vocab: int) -> int:
    """Keep base inputs plus accepted draft inputs; exclude rejected suffix.

    N valid sampled tokens imply N-1 accepted draft tokens. The base input
    predicts the first sampled token and must be kept even on total rejection.
    """
    count = sum(0 <= token < vocab for token in sampled)
    return confirmed_row_count(scheduled, drafts, count)


def confirmed_row_count(scheduled: int, drafts: int, count: int) -> int:
    if drafts:
        return max(0, min(scheduled, scheduled - drafts + max(count - 1, 0)))
    return scheduled


def make_plan(
    training: Any,
    *,
    prompt_len: int,
    max_tokens: int,
    request_id: str,
    global_step: int,
    replica_rank: int,
) -> dict[str, Any]:
    rows = min(
        int(training.get("hidden_state_window_tokens_per_sample", 512)), max_tokens
    )
    offset = 0
    if training.get("hidden_state_window_mode", "front") == "random":
        limit = min(
            int(training.get("hidden_state_random_max_offset") or max_tokens),
            max_tokens,
        )
        rows = min(rows, limit)
        seed = f"{request_id}:{replica_rank}:{global_step if training.get('hidden_state_random_seed_by_step', True) else ''}"
        offset = int.from_bytes(
            hashlib.blake2b(seed.encode(), digest_size=8).digest(), "big"
        ) % (max(limit - rows, 0) + 1)
    return {
        "start": prompt_len - 1 + offset,
        "end": prompt_len - 1 + offset + rows,
        "min_rows": int(training.get("hidden_state_window_min_rows", 1)),
        "global_step": global_step,
        "replica_rank": replica_rank,
        "sample_id": uuid.uuid4().hex,
    }


def capture_root() -> Path:
    return Path(tempfile.gettempdir()) / "verl-speco-rollout-capture"


def plan_directory(plan: dict[str, Any]) -> Path:
    sample_id = plan["sample_id"]
    if (
        not isinstance(sample_id, str)
        or len(sample_id) != 32
        or any(c not in "0123456789abcdef" for c in sample_id)
    ):
        raise ValueError("Invalid capture sample ID")
    return capture_root() / sample_id


def capture_sampled_rows(runner: Any, state: Any, sampler_output: Any) -> None:
    import torch

    batch = runner.input_batch
    selected = []
    for index, request_id in enumerate(batch.req_ids):
        params = runner.requests[request_id].sampling_params
        plan = (getattr(params, "extra_args", None) or {}).get(CAPTURE_KEY)
        if not plan:
            continue
        start = int(batch.num_computed_tokens_cpu[index])
        scheduled = int(state.scheduler_output.num_scheduled_tokens.get(request_id, 0))
        if (
            scheduled > 0
            and start < int(plan["end"])
            and start + scheduled > int(plan["start"])
        ):
            selected.append((index, request_id, plan))
    if not selected:
        return
    if runner.use_async_scheduling:
        raise RuntimeError("vLLM capture does not support async scheduling")
    if not state.aux_hidden_states:
        raise RuntimeError("Target model did not return auxiliary hidden states")
    layer_ids = runner._get_eagle3_aux_layers_from_config()
    if layer_ids is None:
        layer_ids = runner.get_model().get_eagle3_default_aux_hidden_state_layers()
    layer_ids = list(layer_ids)
    if len(layer_ids) != len(state.aux_hidden_states):
        raise RuntimeError("Auxiliary layer metadata does not match model outputs")
    depth = getattr(runner.model_config.hf_config, "num_hidden_layers", None)
    if depth is None or any(
        type(layer) is not int or not 0 <= layer < depth for layer in layer_ids
    ):
        raise ValueError(
            "Capture aux layers must exclude the pre-norm final-layer output"
        )
    for index, request_id, plan in selected:
        directory = plan_directory(plan)
        # The frontend owns this directory. Cancellation removes the live marker.
        if (
            not (directory / "live").exists()
            or (directory / "error").exists()
            or (directory / "skipped").exists()
        ):
            continue
        begin = int(runner.query_start_loc.np[index])
        scheduled = int(state.scheduler_output.num_scheduled_tokens[request_id])
        drafts = len(
            state.scheduler_output.scheduled_spec_decode_tokens.get(request_id, [])
        )
        start = int(batch.num_computed_tokens_cpu[index])
        lo, hi = (
            max(start, int(plan["start"])),
            min(start + scheduled, int(plan["end"])),
        )
        if lo >= hi:
            continue
        selection = slice(begin + lo - start, begin + hi - start)
        parts = [h[selection] for h in state.aux_hidden_states]
        if plan["layout"].endswith("plus_last"):
            parts.append(state.hidden_states[selection])
        # Reserve before allocating snapshots/pinned destinations. Admission is
        # nonblocking; an overloaded sample is explicitly discarded as a whole.
        # Include the GPU snapshot, pinned destination and writer-side compact
        # accepted-row payload, which can coexist during serialization.
        size = sum(h.numel() * h.element_size() for h in parts) * 3
        size += (hi - lo) * (runner.input_ids.gpu.element_size() + 8) * 3
        size += 16  # owned device/host acceptance-count scalar
        queue = export_queue()
        if not queue.reserve(size):
            (directory / "skipped").touch()
            logger.warning(
                "Skipped capture sample %s: export capacity exhausted",
                plan["sample_id"],
            )
            continue
        done = None
        stream = None
        try:
            # cat and clone run on the forward stream, so next graph execution
            # cannot overwrite their source before these snapshots are complete.
            snapshot = torch.cat(parts, dim=-1).detach()
            token_snapshot = runner.input_ids.gpu[selection].detach().clone()
            sampled = sampler_output.sampled_token_ids[index]
            count_snapshot = ((sampled >= 0) & (sampled < batch.vocab_size)).sum()
            if snapshot.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(snapshot.device))
                stream = getattr(runner, "_speco_capture_copy_stream", None)
                if stream is None:
                    stream = runner._speco_capture_copy_stream = torch.cuda.Stream(
                        device=snapshot.device
                    )
                hidden = torch.empty_like(snapshot, device="cpu", pin_memory=True)
                ids = torch.empty_like(token_snapshot, device="cpu", pin_memory=True)
                count = torch.empty_like(count_snapshot, device="cpu", pin_memory=True)
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    hidden.copy_(snapshot, non_blocking=True)
                    ids.copy_(token_snapshot, non_blocking=True)
                    count.copy_(count_snapshot, non_blocking=True)
                    done = torch.cuda.Event()
                    done.record(stream)
                snapshot.record_stream(stream)
                token_snapshot.record_stream(stream)
                count_snapshot.record_stream(stream)
            else:
                hidden, ids = snapshot, token_snapshot
                count = count_snapshot
            payload = {
                "hidden_states": hidden,
                "token_ids": ids,
                "layer_ids": layer_ids,
                "layout": plan["layout"],
                "last_hidden_norm": "post_norm",
                "positions": torch.arange(lo, hi),
                "sample_id": plan["sample_id"],
            }

            def finish_copy(
                event=done,
                data=payload,
                accepted=count,
                offset=start,
                total=scheduled,
                draft_count=drafts,
            ):
                if event is not None:
                    event.synchronize()
                # Read the scalar only after D2H completes, outside the runner.
                keep = confirmed_row_count(total, draft_count, int(accepted))
                valid = data["positions"] < offset + keep
                for field in ("hidden_states", "token_ids", "positions"):
                    data[field] = data[field][valid]

            queue.submit_reserved(
                directory,
                payload,
                size=size,
                wait=finish_copy,
                owners=(snapshot, token_snapshot, count_snapshot, hidden, ids, count),
            )
            if logger.isEnabledFor(logging.DEBUG):
                d2h_bytes = (
                    sum(t.numel() * t.element_size() for t in (hidden, ids, count))
                    if done is not None
                    else 0
                )
                logger.debug(
                    "vLLM capture chunk sample=%s window=[%s,%s) d2h_bytes=%s",
                    plan["sample_id"],
                    lo,
                    hi,
                    d2h_bytes,
                )
        except BaseException:
            # Copies may have been launched before a submission error. Wait
            # before dropping pinned buffers and releasing their reservation.
            try:
                if done is not None:
                    done.synchronize()
                elif stream is not None:
                    stream.synchronize()
            finally:
                queue.release(size)
            if (directory / "live").exists():
                raise
            continue


def install_runner_capture() -> None:
    """Install before model loading in each vLLM worker process."""
    from importlib.metadata import version
    from packaging.version import Version

    if Version(version("vllm")).base_version != "0.22.0":
        raise RuntimeError("Rollout capture currently requires vLLM 0.22.0 V1")
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_speco_capture_installed", False):
        return
    original_sample_tokens = GPUModelRunner.sample_tokens
    original_sample = GPUModelRunner._sample
    original_init = GPUModelRunner.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = self.vllm_config
        if getattr(config.speculative_config, "method", None) not in {
            "eagle3",
            "dflash",
        }:
            raise ValueError(
                "Direct capture requires a real EAGLE3/DFlash drafter; extractor mode conflicts"
            )
        model_type = getattr(config.model_config.hf_config, "model_type", "")
        if model_type not in {"llama", "qwen2", "qwen3"}:
            raise ValueError(
                "vLLM capture currently supports Llama/Qwen2/Qwen3 text targets"
            )
        parallel = config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.use_ubatching
        ):
            raise ValueError("vLLM capture requires TP1/PP1/DP1 without microbatching")
        if self.use_async_scheduling or config.cache_config.enable_prefix_caching:
            raise ValueError(
                "vLLM capture requires sync scheduling and no prefix cache"
            )
        if self.device.type != "cuda" or self.supports_mm_inputs:
            raise ValueError("vLLM capture requires CUDA and a text-only target")
        if not self.use_aux_hidden_state_outputs:
            raise ValueError("vLLM capture requires an auxiliary-hidden-state drafter")

    @functools.wraps(original_sample_tokens)
    def sample_tokens(self, *args, **kwargs):
        self._speco_capture_state = self.execute_model_state
        try:
            return original_sample_tokens(self, *args, **kwargs)
        finally:
            self._speco_capture_state = None

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        state = getattr(self, "_speco_capture_state", None)
        if state is not None:
            try:
                capture_sampled_rows(self, state, result)
            except Exception:
                # Feature export must not take down unrelated engine requests.
                # Mark selected requests invalid; the frontend fails closed.
                logger.exception("vLLM rollout feature capture failed")
                for request_id in self.input_batch.req_ids:
                    params = self.requests[request_id].sampling_params
                    plan = (getattr(params, "extra_args", None) or {}).get(CAPTURE_KEY)
                    if plan:
                        try:
                            (plan_directory(plan) / "error").touch()
                        except OSError:
                            logger.warning("Could not persist capture failure marker")
        return result

    GPUModelRunner.sample_tokens = sample_tokens
    GPUModelRunner._sample = sample
    GPUModelRunner.__init__ = init
    GPUModelRunner._speco_capture_installed = True


def load_sample(
    plan: dict[str, Any], prompt: list[int], response: list[int]
) -> dict[str, Any] | None:
    """Validate position coverage and tokens before entering the shared trainer."""
    import torch

    directory = plan_directory(plan)
    from verl_speco.producer.feature_export import (
        wait_for_exports,
        rollout_feature_fields,
    )

    if (directory / "error").exists():
        raise RuntimeError("vLLM worker reported a hidden-state export failure")
    end = min(int(plan["end"]), len(prompt) + len(response) - 1)
    start = int(plan["start"])
    if end - start < int(plan["min_rows"]) or end <= start:
        return None
    wait_started = time.perf_counter()
    wait_for_exports(directory)
    logger.debug(
        "vLLM capture export wait sample=%s seconds=%.6f",
        plan["sample_id"],
        time.perf_counter() - wait_started,
    )
    if (directory / "skipped").exists():
        return None
    if (directory / "error").exists():
        raise RuntimeError("vLLM worker reported a hidden-state export failure")
    chunks = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in sorted(directory.glob("*.pt"))
    ]
    rows: dict[int, tuple[Any, int]] = {}
    layer_ids = None
    for chunk in chunks:
        if chunk["sample_id"] != plan["sample_id"]:
            raise RuntimeError("Capture sample identity mismatch")
        if (
            chunk["layout"] != plan["layout"]
            or chunk["last_hidden_norm"] != "post_norm"
        ):
            raise RuntimeError("Capture hidden layout/norm mismatch")
        if layer_ids is not None and layer_ids != chunk["layer_ids"]:
            raise RuntimeError("Capture auxiliary layers changed during request")
        layer_ids = chunk["layer_ids"]
        for p, h, token in zip(
            chunk["positions"].tolist(),
            chunk["hidden_states"],
            chunk["token_ids"].tolist(),
            strict=True,
        ):
            if p in rows:
                # Fail closed rather than silently choosing stale replay data.
                if rows[p][1] != token or not torch.equal(rows[p][0], h):
                    raise RuntimeError("Conflicting recomputed capture position")
            rows[p] = (h, token)
    tokens = prompt + response
    positions = list(range(start, end))
    if any(p not in rows or rows[p][1] != tokens[p] for p in positions):
        raise RuntimeError(
            "vLLM rollout capture missing rows or token alignment mismatch"
        )
    return {
        **rollout_feature_fields(
            torch.tensor(prompt),
            torch.tensor(response),
            torch.stack([rows[p][0] for p in positions]),
            torch.tensor(positions),
        ),
        "hidden_position_start": start,
        "hidden_position_end": end,
        "hidden_states_layout": plan["layout"],
        "hidden_source": "vllm_rollout",
        "hidden_layer_ids": layer_ids,
        "last_hidden_state_norm": "post_norm",
        "global_step": plan["global_step"],
        "replica_rank": plan["replica_rank"],
        "_speco_sample_id": plan["sample_id"],
    }


def cleanup(plan: dict[str, Any]) -> None:
    directory = plan_directory(plan)
    (directory / "live").unlink(missing_ok=True)
    shutil.rmtree(directory, ignore_errors=True)


async def generate_with_capture(
    server: Any,
    generate: Any,
    args: tuple,
    kwargs: dict,
    drafter: dict,
    *,
    skip: bool = False,
) -> Any:
    """Frontend adapter; expensive imports and file allocation are opt-in."""
    if not enabled(drafter):
        return await generate(*args, **kwargs)
    if skip:
        logger.debug(
            "vLLM capture skipped reason=request_skip step=%s replica=%s",
            server.global_steps,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    import inspect
    from verl_speco.trainer.scheduler import (
        DrafterCollectionContext,
        DrafterCollectionSource,
        DrafterScheduleConfig,
        DrafterScheduler,
    )
    from verl_speco.integration.oldlogprob_layer_ids import (
        resolve_drafter_hidden_states_layout,
    )

    training = drafter["training"]
    step = int(server.global_steps)
    scheduler = DrafterScheduler()
    decision = scheduler.plan_collection(
        DrafterCollectionContext(
            global_step=step,
            source=DrafterCollectionSource.VLLM,
            drafter_enabled=True,
            source_enabled=True,
            validation=False,
        ),
        DrafterScheduleConfig.from_mapping(training),
    )
    if not decision.collect:
        logger.debug(
            "vLLM capture skipped reason=%s step=%s replica=%s",
            decision.reason,
            step,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    bound = inspect.signature(generate).bind(*args, **kwargs)
    values = bound.arguments
    prompt = values.get("prompt_ids")
    params = values.get("sampling_params")
    request_id = str(values.get("request_id", ""))
    if not isinstance(prompt, list) or not prompt or not isinstance(params, dict):
        raise RuntimeError(
            "Unsupported verl vLLM generate signature for rollout capture"
        )
    if int(params.get("n", 1)) != 1 or any(
        values.get(k) is not None for k in ("image_data", "video_data", "audio_data")
    ):
        raise ValueError("vLLM capture requires text-only n=1 requests")
    maximum = int(
        params.get(
            "max_tokens", params.get("max_new_tokens", server.config.response_length)
        )
    )
    plan = make_plan(
        training,
        prompt_len=len(prompt),
        max_tokens=maximum,
        request_id=request_id,
        global_step=step,
        replica_rank=int(server.replica_rank),
    )
    if plan["end"] - plan["start"] < plan["min_rows"]:
        logger.debug(
            "vLLM capture skipped reason=window_too_short request=%s step=%s planned_rows=%s min_rows=%s",
            request_id,
            step,
            plan["end"] - plan["start"],
            plan["min_rows"],
        )
        return await generate(*args, **kwargs)
    # Replica-local admission happens before any await, so concurrent requests
    # cannot exceed the configured per-step sample/token budgets.
    budget = getattr(server, "_speco_capture_budget", None)
    if budget is None or budget[0] != step:
        budget = server._speco_capture_budget = [step, 0, 0]
    sample_limit = int(training.get("max_collect_samples_per_step_per_replica", 16))
    token_limit = int(training.get("max_collect_tokens_per_step_per_replica", 16384))
    rows = plan["end"] - plan["start"]
    fraction = (
        int.from_bytes(
            hashlib.blake2b(request_id.encode(), digest_size=8).digest(), "big"
        )
        / 2**64
    )
    reason = None
    if sample_limit > 0 and budget[1] >= sample_limit:
        reason = "sample_budget"
    elif token_limit > 0 and budget[2] + rows > token_limit:
        reason = "token_budget"
    elif fraction >= float(training.get("collection_sample_rate", 1.0)):
        reason = "sample_rate"
    if reason:
        logger.debug(
            "vLLM capture skipped reason=%s request=%s step=%s replica=%s",
            reason,
            request_id,
            step,
            server.replica_rank,
        )
        return await generate(*args, **kwargs)
    budget[1] += 1
    budget[2] += rows
    plan["layout"] = resolve_drafter_hidden_states_layout(
        drafter["speculative_algorithm"], training
    )
    directory = plan_directory(plan)
    extra = dict(params.get("extra_args") or {})
    if CAPTURE_KEY in extra:
        raise ValueError("speco_rollout_capture is reserved for the rollout adapter")
    extra[CAPTURE_KEY] = plan
    values["sampling_params"] = {**params, "extra_args": extra}
    directory.mkdir(parents=True, exist_ok=False)
    try:
        (directory / "live").touch()
        logger.debug(
            "vLLM capture admitted request=%s sample=%s step=%s replica=%s window=[%s,%s) min_rows=%s layout=%s",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
            plan["start"],
            plan["end"],
            plan["min_rows"],
            plan["layout"],
        )
        output = await generate(*bound.args, **bound.kwargs)
        if getattr(output, "stop_reason", None) == "aborted":
            logger.debug(
                "vLLM capture skipped reason=aborted request=%s sample=%s",
                request_id,
                plan["sample_id"],
            )
            return output
        if int(server.global_steps) != step:
            raise RuntimeError("Target serving version changed during capture")
        import asyncio

        load_started = time.perf_counter()
        sample = await asyncio.to_thread(
            load_sample, plan, prompt, list(output.token_ids)
        )
        output.extra_fields["_speco_capture_skipped_capacity"] = int(
            (directory / "skipped").exists()
        )
        output.extra_fields["_speco_capture_rows"] = (
            int(sample["hidden_states"].shape[1]) if sample is not None else 0
        )
        logger.debug(
            "vLLM capture result request=%s sample=%s step=%s replica=%s reason=%s rows=%s response_tokens=%s load_seconds=%.6f",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
            "collected"
            if sample is not None
            else (
                "capacity"
                if output.extra_fields["_speco_capture_skipped_capacity"]
                else "short_response"
            ),
            output.extra_fields["_speco_capture_rows"],
            len(output.token_ids),
            time.perf_counter() - load_started,
        )
        if sample is not None:
            from verl_speco.integration.rollout_idle_events import (
                emit_rollout_drafter_sample,
            )

            name = ((training.get("scheduler") or {}).get("idle_worker") or {}).get(
                "event_bus_name"
            )
            emitted = emit_rollout_drafter_sample(
                name,
                sample,
                sample_id=plan["sample_id"],
                replica_rank=plan["replica_rank"],
                global_step=step,
            )
            sample["_speco_direct_event_emitted"] = emitted
            output.extra_fields["drafter_sample"] = sample
        return output
    except Exception:
        logger.exception(
            "vLLM capture failed request=%s sample=%s step=%s replica=%s",
            request_id,
            plan["sample_id"],
            step,
            server.replica_rank,
        )
        raise
    finally:
        cleanup(plan)
