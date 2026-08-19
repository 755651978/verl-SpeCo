# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone torchrun training loop for SPECO draft models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from omegaconf import OmegaConf, open_dict
from verl.utils.device import get_device_name, get_torch_device

from verl_speco.backends.factory import build_trainer_backend
from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.trainer.draft_dataset import (
    DraftFeatureDataLoader,
    DraftFeatureDataLoaderConfig,
)
from verl_speco.trainer.feature_store import build_feature_store_from_config
from verl_speco.trainer.standalone_checkpoint import rewrite_standalone_runtime_config

logger = logging.getLogger(__name__)


def _should_log_batch_progress(attempted_batches: int) -> bool:
    return attempted_batches <= 3 or attempted_batches % 100 == 0


def _is_out_of_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    if "out of memory" in message or "oom" in message:
        return True
    return error.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}


def run_standalone_draft_training(config) -> dict[str, Any]:
    """Run independent draft training from a feature store."""
    return asyncio.run(_run_standalone_draft_training_async(config))


async def _run_standalone_draft_training_async(config) -> dict[str, Any]:
    rank, local_rank, world_size = _init_distributed()
    logger.info(
        "[standalone rank=%s] distributed runtime initialized local_rank=%s world_size=%s",
        rank,
        local_rank,
        world_size,
    )
    draft_config = config.actor_rollout_ref
    drafter_cfg = draft_config.rollout.drafter
    training_cfg = drafter_cfg.training
    feature_store_cfg = training_cfg.feature_store
    feature_store_type = (
        str(feature_store_cfg.get("type", "torch_shard") or "torch_shard")
        .strip()
        .lower()
    )
    training_mode = (
        str(training_cfg.get("mode", "offline") or "offline").strip().lower()
    )
    replay_feature_store_types = {"token_replay", "jsonl_token_replay", "jsonl"}
    if not feature_store_cfg.get("path"):
        raise ValueError(
            "actor_rollout_ref.rollout.drafter.training.feature_store.path is required"
        )
    if feature_store_type in replay_feature_store_types and training_mode != "offline":
        raise ValueError(
            f"feature_store.type={feature_store_type} is supported only by "
            "standalone training.mode=offline"
        )
    _disable_standalone_sequence_parallel(draft_config)

    _configure_device(local_rank)
    backend = _build_backend(draft_config)
    setattr(backend, "enable_standalone_training_metrics", True)
    training_device_mesh = _build_training_device_mesh(draft_config, world_size)
    trainer = DrafterBaseTrainer(
        config=draft_config,
        world_size=world_size,
        # Standalone ranks form one training replica. Keep rollout_dp_rank at
        # zero on every rank so all ranks participate in optimizer DCP while
        # _is_checkpoint_leader still selects SP rank zero for metadata/model IO.
        rollout_dp_rank=0,
        training_device_mesh=training_device_mesh,
        training_process_group=(
            None
            if training_device_mesh is not None
            else dist.group.WORLD
            if dist.is_initialized() and world_size > 1
            else None
        ),
        data_parallel_process_group=None,
        backend=backend,
    )
    max_steps = int(training_cfg.get("max_steps", training_cfg.get("step", 1000)) or 0)
    save_interval = int(training_cfg.get("save_interval_steps", 0) or 0)
    successful_steps = 0
    initial_optimizer_step = 0
    optimizer_step = 0
    attempted_batches = 0
    last_save_result: dict[str, Any] | None = None
    last_saved_step = 0
    store = None
    feature_replayer = None
    current_stage = "activate_training_model"
    try:
        stage_started = time.perf_counter()
        logger.info(
            "[standalone rank=%s] activating drafter model algorithm=%s",
            rank,
            drafter_cfg.speculative_algorithm,
        )
        activated = await trainer.activate_training_model()
        if not activated:
            raise RuntimeError(
                f"Failed to activate standalone drafter trainer on rank={rank}"
            )
        logger.info(
            "[standalone rank=%s] drafter model activated elapsed=%.3fs",
            rank,
            time.perf_counter() - stage_started,
        )
        initial_optimizer_step = int(trainer.optimizer_steps_total)
        optimizer_step = initial_optimizer_step
        last_saved_step = optimizer_step

        current_stage = "open_feature_store"
        stage_started = time.perf_counter()
        logger.info(
            "[standalone rank=%s] opening feature store type=%s path=%s",
            rank,
            feature_store_type,
            feature_store_cfg.get("path"),
        )
        if feature_store_type in {"jsonl_token_replay", "jsonl"} and not (
            feature_store_cfg.get("tokenizer_path")
        ):
            tokenizer_path = draft_config.actor_rollout_ref.model.path
            try:
                feature_store_cfg.tokenizer_path = tokenizer_path
            except AttributeError:
                feature_store_cfg["tokenizer_path"] = tokenizer_path
        store = build_feature_store_from_config(feature_store_cfg, read_only=True)
        logger.info(
            "[standalone rank=%s] feature store opened elapsed=%.3fs",
            rank,
            time.perf_counter() - stage_started,
        )
        if feature_store_type in replay_feature_store_types:
            # Keep the large target model entirely outside online training imports
            # and lifetime. The standalone loop materializes ordinary feature
            # samples before handing them to the shared trainer.
            from verl_speco.trainer.target_feature_replay import (
                TargetFeatureReplayer,
            )

            current_stage = "initialize_target_feature_replayer"
            stage_started = time.perf_counter()
            logger.info(
                "[standalone rank=%s] initializing target feature replayer",
                rank,
            )
            feature_replayer = TargetFeatureReplayer(
                config,
                rank=rank,
                world_size=world_size,
                device=trainer.runtime_device,
            )
            logger.info(
                "[standalone rank=%s] target feature replayer initialized "
                "backend=%s elapsed=%.3fs",
                rank,
                feature_replayer.backend,
                time.perf_counter() - stage_started,
            )
        current_stage = "create_dataloader"
        loader = DraftFeatureDataLoader(
            store,
            DraftFeatureDataLoaderConfig(
                batch_size=int(training_cfg.get("batch_size_per_gpu", 4)),
                rank=rank,
                world_size=world_size,
                shuffle=bool(feature_store_cfg.get("shuffle", True)),
                repeat=bool(feature_store_cfg.get("repeat", True)),
                seed=int(training_cfg.get("seed", 0) or 0),
            ),
        )
        logger.info(
            "[standalone rank=%s] dataloader ready batch_size_per_gpu=%s "
            "shuffle=%s repeat=%s",
            rank,
            int(training_cfg.get("batch_size_per_gpu", 4)),
            bool(feature_store_cfg.get("shuffle", True)),
            bool(feature_store_cfg.get("repeat", True)),
        )
        for samples in loader:
            if max_steps > 0 and successful_steps >= max_steps:
                break
            step_started = time.perf_counter()
            attempted_batches += 1
            log_batch_progress = _should_log_batch_progress(attempted_batches)
            if log_batch_progress:
                logger.info(
                    "[standalone rank=%s] batch=%s loaded samples=%s "
                    "successful_steps=%s",
                    rank,
                    attempted_batches,
                    len(samples),
                    successful_steps,
                )
            current_stage = "materialize_target_features"
            if feature_replayer is not None:
                materialize_started = time.perf_counter()
                if log_batch_progress:
                    logger.info(
                        "[standalone rank=%s] batch=%s materializing target features "
                        "backend=%s",
                        rank,
                        attempted_batches,
                        feature_replayer.backend,
                    )
                materialized_samples = feature_replayer.materialize(samples)
                if log_batch_progress:
                    logger.info(
                        "[standalone rank=%s] batch=%s target features materialized "
                        "samples=%s elapsed=%.3fs",
                        rank,
                        attempted_batches,
                        len(materialized_samples),
                        time.perf_counter() - materialize_started,
                    )
            else:
                materialized_samples = samples
            current_stage = "prepare_training_batch"
            batch = trainer.prepare_training_batch_from_samples(
                cast(list[Any], materialized_samples),
                step=optimizer_step,
            )
            has_batch = batch is not None
            current_stage = "synchronize_batch_readiness"
            if not _all_ranks_true(has_batch, trainer.runtime_device):
                if rank == 0:
                    logger.warning(
                        "Skipping standalone drafter batch: at least one rank has no valid batch"
                    )
                continue
            if batch is None:
                continue
            trainer.reset_training_metrics()
            current_stage = "training_step"
            if log_batch_progress:
                logger.info(
                    "[standalone rank=%s] batch=%s starting drafter training step "
                    "optimizer_step=%s",
                    rank,
                    attempted_batches,
                    optimizer_step,
                )
            ok = await trainer.training_step_from_batch(batch, optimizer_step)
            step_error = getattr(trainer, "last_standalone_training_error", None)
            if step_error is not None and _is_out_of_memory_error(step_error):
                raise RuntimeError(
                    "Standalone drafter training hit an unrecoverable OOM during "
                    f"batch={attempted_batches} optimizer_step={optimizer_step}. "
                    "Reduce batch_size_per_gpu, feature_store.max_seq_len, "
                    "dspark_num_anchors/block_size or disable DSpark L1 loss."
                ) from step_error
            current_stage = "synchronize_training_step"
            if not _all_ranks_true(ok, trainer.runtime_device):
                continue
            successful_steps += 1
            optimizer_step = int(trainer.optimizer_steps_total)
            if optimizer_step <= initial_optimizer_step:
                optimizer_step = initial_optimizer_step + successful_steps
            step_metrics = _standalone_step_metrics(
                trainer,
                successful_steps=successful_steps,
                attempted_batches=attempted_batches,
                step_elapsed_sec=time.perf_counter() - step_started,
            )
            if feature_replayer is not None:
                step_metrics.update(feature_replayer.metrics())
            _log_standalone_step_metrics(step_metrics, rank=rank)
            if save_interval > 0 and optimizer_step % save_interval == 0:
                current_stage = "save_checkpoint"
                last_save_result = _save_standalone_checkpoint(trainer, optimizer_step)
                if _sync_any_rank_saved_checkpoint(last_save_result.get("saved")):
                    last_saved_step = optimizer_step
                _barrier()
            current_stage = "load_next_batch"
        final_save = bool(training_cfg.get("save_final_checkpoint", True))
        if final_save and successful_steps > 0 and optimizer_step != last_saved_step:
            current_stage = "save_final_checkpoint"
            last_save_result = _save_standalone_checkpoint(
                trainer, optimizer_step, wait=True
            )
            _barrier()
    except Exception:
        logger.exception(
            "[standalone rank=%s] training failed stage=%s attempted_batches=%s "
            "successful_steps=%s optimizer_step=%s",
            rank,
            current_stage,
            attempted_batches,
            successful_steps,
            optimizer_step,
        )
        raise
    finally:
        logger.info(
            "[standalone rank=%s] cleanup starting stage=%s attempted_batches=%s "
            "successful_steps=%s",
            rank,
            current_stage,
            attempted_batches,
            successful_steps,
        )
        if store is not None:
            store.close()
        if feature_replayer is not None:
            feature_replayer.close()
        logger.info("[standalone rank=%s] cleaning trainer resources", rank)
        await trainer.cleanup_training(clear_data=True)
        if dist.is_initialized():
            logger.info(
                "[standalone rank=%s] entering final process-group barrier", rank
            )
            dist.barrier()
            logger.info(
                "[standalone rank=%s] final process-group barrier complete", rank
            )
            dist.destroy_process_group()
        logger.info("[standalone rank=%s] cleanup complete", rank)

    return {
        "rank": rank,
        "world_size": world_size,
        "attempted_batches": attempted_batches,
        "successful_steps": successful_steps,
        "initial_optimizer_step": initial_optimizer_step,
        "optimizer_steps_total": optimizer_step,
        "last_save": last_save_result,
    }


def _build_backend(draft_config):
    return build_trainer_backend(draft_config, draft_config.model)


def _save_standalone_checkpoint(
    trainer: DrafterBaseTrainer, step: int, *, wait: bool = False
) -> dict[str, Any]:
    save_checkpoint = getattr(trainer, "save_checkpoint", None)
    if callable(save_checkpoint):
        result = save_checkpoint(int(step), wait=wait)
        checkpoint_path = result.get("path")
        is_export_leader = result.get("reason") in {"saved", "scheduled"}
        if result.get("saved") and checkpoint_path and is_export_leader:
            if wait:
                _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
            else:
                future = getattr(trainer, "_pending_full_checkpoint_future", None)
                if future is not None:
                    future.add_done_callback(
                        lambda completed: _finalize_standalone_checkpoint(
                            trainer,
                            checkpoint_path,
                            completed,
                        )
                    )
        return result

    # Keep the small PR #13 test double and older trainer adapters usable.
    checkpoint_dir = getattr(trainer, "checkpoint_dir", None)
    if not checkpoint_dir:
        return {"saved": False, "reason": "missing_checkpoint_dir"}
    checkpoint_path = os.path.join(checkpoint_dir, f"draft_step_{int(step)}")
    pending_full_checkpoint = getattr(trainer, "_pending_full_checkpoint_future", None)
    pending_done = getattr(pending_full_checkpoint, "done", None)
    if callable(pending_done) and not pending_done():
        return {
            "saved": False,
            "path": checkpoint_path,
            "reason": "previous_save_running",
        }

    save_async = getattr(trainer, "_save_checkpoint_async", None)
    if not callable(save_async):
        return {
            "saved": False,
            "path": checkpoint_path,
            "reason": "unsupported_trainer",
        }
    future = save_async(int(step))
    if future is not None and wait:
        future.result()
        trainer._pending_full_checkpoint_future = None
        _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)
    elif future is not None:
        future.add_done_callback(
            lambda completed: _rewrite_standalone_block_runtime_config(
                trainer,
                checkpoint_path,
                completed,
            )
        )
    return {
        "saved": future is not None,
        "path": checkpoint_path,
        "reason": (
            "saved"
            if future is not None and wait
            else "scheduled"
            if future is not None
            else "not_checkpoint_leader"
        ),
    }


def _load_tensor_from_safetensors(
    path: str, keys: tuple[str, ...]
) -> tuple[str, torch.Tensor] | None:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            available_keys = set(f.keys())
            for key in keys:
                if key in available_keys:
                    return key, f.get_tensor(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load any of %s from %s: %s", keys, path, exc)
    return None


def _finalize_standalone_checkpoint(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future,
) -> None:
    try:
        completed_future.result()
    except Exception:
        _rewrite_standalone_block_runtime_config(
            trainer, checkpoint_path, completed_future
        )
        return

    _rewrite_standalone_block_runtime_config(trainer, checkpoint_path)


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_tensor_from_torch(
    path: str, keys: tuple[str, ...]
) -> tuple[str, torch.Tensor] | None:
    try:
        state = _torch_load_cpu(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load any of %s from %s: %s", keys, path, exc)
        return None
    if isinstance(state, dict):
        for key in keys:
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                return key, value
    return None


def _target_model_path_for_lm_head(trainer: DrafterBaseTrainer) -> str | None:
    model_path = getattr(getattr(trainer, "config", None), "model", None)
    model_path = getattr(model_path, "path", None)
    if not model_path:
        return None
    return os.fspath(model_path)


def _model_ties_word_embeddings(model_path: str | None) -> bool:
    if not model_path:
        return False
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read model config %s for tied embedding check: %s",
            config_path,
            exc,
        )
        return False
    return bool(isinstance(config, dict) and config.get("tie_word_embeddings") is True)


def _load_lm_head_weight(
    model_path: str | None,
    *,
    allow_tied_embedding: bool = False,
) -> tuple[str, torch.Tensor] | None:
    if not model_path:
        return None

    keys = (
        ("lm_head.weight", "model.embed_tokens.weight")
        if allow_tied_embedding
        else ("lm_head.weight",)
    )
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(model_path, index_name)
        if not os.path.exists(index_path):
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read weight index %s: %s", index_path, exc)
            continue
        if not isinstance(weight_map, dict):
            continue
        for key in keys:
            shard_name = weight_map.get(key)
            if not shard_name:
                continue
            shard_path = os.path.join(model_path, os.fspath(shard_name))
            if index_name.endswith(".safetensors.index.json"):
                loaded = _load_tensor_from_safetensors(shard_path, (key,))
            else:
                loaded = _load_tensor_from_torch(shard_path, (key,))
            if loaded is not None:
                return loaded

    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        loaded = _load_tensor_from_safetensors(safetensors_path, keys)
        if loaded is not None:
            return loaded

    torch_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(torch_path):
        return _load_tensor_from_torch(torch_path, keys)

    return None


def _append_lm_head_to_safetensors_index(
    checkpoint_path: str, lm_head_weight: torch.Tensor
) -> bool:
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        from safetensors.torch import save_file

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.setdefault("weight_map", {})
        if not isinstance(weight_map, dict):
            logger.warning(
                "Cannot append lm_head.weight to %s: expected weight_map object",
                index_path,
            )
            return True
        if "lm_head.weight" in weight_map:
            return True

        shard_name = "model-lm-head.safetensors"
        save_file(
            {"lm_head.weight": lm_head_weight},
            os.path.join(checkpoint_path, shard_name),
        )
        weight_map["lm_head.weight"] = shard_name
        metadata = index_data.setdefault("metadata", {})
        if isinstance(metadata, dict) and "total_size" in metadata:
            metadata["total_size"] = int(metadata["total_size"]) + (
                lm_head_weight.numel() * lm_head_weight.element_size()
            )
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(
            "Added lm_head.weight to standalone sharded checkpoint %s", index_path
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to append lm_head.weight to sharded checkpoint %s: %s",
            index_path,
            exc,
        )
    return True


def _append_lm_head_to_torch_index(
    checkpoint_path: str, lm_head_weight: torch.Tensor
) -> bool:
    index_path = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.setdefault("weight_map", {})
        if not isinstance(weight_map, dict):
            logger.warning(
                "Cannot append lm_head.weight to %s: expected weight_map object",
                index_path,
            )
            return True
        if "lm_head.weight" in weight_map:
            return True

        shard_name = "pytorch_model-lm-head.bin"
        torch.save(
            {"lm_head.weight": lm_head_weight},
            os.path.join(checkpoint_path, shard_name),
        )
        weight_map["lm_head.weight"] = shard_name
        metadata = index_data.setdefault("metadata", {})
        if isinstance(metadata, dict) and "total_size" in metadata:
            metadata["total_size"] = int(metadata["total_size"]) + (
                lm_head_weight.numel() * lm_head_weight.element_size()
            )
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(
            "Added lm_head.weight to standalone sharded checkpoint %s", index_path
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to append lm_head.weight to sharded checkpoint %s: %s",
            index_path,
            exc,
        )
    return True


def _append_lm_head_weight_if_missing(
    checkpoint_path: str,
    source_model_path: str | None,
    target_model_path: str | None,
    *,
    allow_target_fallback: bool,
) -> None:
    loaded = _load_lm_head_weight(source_model_path)
    source_label = source_model_path
    if loaded is None and allow_target_fallback:
        loaded = _load_lm_head_weight(target_model_path)
        source_label = target_model_path
        if loaded is None and _model_ties_word_embeddings(target_model_path):
            loaded = _load_lm_head_weight(target_model_path, allow_tied_embedding=True)
    if loaded is None:
        if allow_target_fallback:
            logger.warning(
                "Standalone DSpark checkpoint export could not find lm_head.weight in source %s, "
                "or target lm_head.weight / tied model.embed_tokens.weight in target %s",
                source_model_path,
                target_model_path,
            )
        return
    loaded_key, lm_head_weight = loaded
    lm_head_weight = lm_head_weight.detach().cpu()
    logger.info(
        "Using %s from %s as standalone drafter lm_head.weight",
        loaded_key,
        source_label,
    )

    if _append_lm_head_to_safetensors_index(checkpoint_path, lm_head_weight):
        return
    if _append_lm_head_to_torch_index(checkpoint_path, lm_head_weight):
        return

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file, save_file

            state = load_file(safetensors_path, device="cpu")
            if "lm_head.weight" in state:
                return
            state["lm_head.weight"] = lm_head_weight
            save_file(state, safetensors_path)
            logger.info(
                "Added lm_head.weight to standalone checkpoint %s", safetensors_path
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to append lm_head.weight to %s: %s", safetensors_path, exc
            )
        return

    torch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(torch_path):
        try:
            state = _torch_load_cpu(torch_path)
            if not isinstance(state, dict):
                logger.warning(
                    "Cannot append lm_head.weight to %s: expected dict state",
                    torch_path,
                )
                return
            if "lm_head.weight" in state:
                return
            state["lm_head.weight"] = lm_head_weight
            torch.save(state, torch_path)
            logger.info("Added lm_head.weight to standalone checkpoint %s", torch_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append lm_head.weight to %s: %s", torch_path, exc)
        return

    logger.warning(
        "Standalone checkpoint export found no model.safetensors or pytorch_model.bin under %s",
        checkpoint_path,
    )


def _rewrite_standalone_block_runtime_config(
    trainer: DrafterBaseTrainer,
    checkpoint_path: str,
    completed_future=None,
) -> None:
    source_model_path = rewrite_standalone_runtime_config(
        trainer, checkpoint_path, completed_future
    )
    backend_type = getattr(getattr(trainer, "backend", None), "model_type", None)
    if backend_type == "dspark":
        _append_lm_head_weight_if_missing(
            checkpoint_path,
            source_model_path,
            _target_model_path_for_lm_head(trainer),
            allow_target_fallback=True,
        )
    elif backend_type == "dflash":
        _append_lm_head_weight_if_missing(
            checkpoint_path,
            source_model_path,
            None,
            allow_target_fallback=False,
        )


def _disable_standalone_sequence_parallel(draft_config) -> None:
    rollout_cfg = draft_config.rollout
    rollout_tp_size = int(rollout_cfg.get("tensor_model_parallel_size", 1) or 1)
    if rollout_tp_size <= 1:
        return
    logger.warning(
        "Standalone draft training disables Ulysses sequence parallelism: "
        "actor_rollout_ref.rollout.tensor_model_parallel_size=%s is treated as 1 for offline drafter training",
        rollout_tp_size,
    )
    with open_dict(rollout_cfg):
        rollout_cfg.tensor_model_parallel_size = 1


def _build_training_device_mesh(draft_config, world_size: int) -> DeviceMesh | None:
    if world_size <= 1 or not dist.is_initialized():
        return None
    strategy = str(
        draft_config.actor.get("strategy", "") if hasattr(draft_config, "actor") else ""
    ).lower()
    if strategy != "fsdp2":
        return None
    return DeviceMesh(
        device_type=get_device_name(),
        mesh=torch.arange(world_size, dtype=torch.int64).reshape(1, world_size),
        mesh_dim_names=("dp", "sp"),
    )


def _block_metric_prefix(trainer: DrafterBaseTrainer) -> str | None:
    model_type = str(getattr(getattr(trainer, "backend", None), "model_type", "") or "")
    if model_type in {"dflash", "dspark", "eagle3"}:
        return model_type
    return None


def _current_learning_rate(trainer: DrafterBaseTrainer) -> float:
    optimizer = getattr(trainer, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not param_groups:
        return 0.0
    return float(param_groups[0].get("lr", 0.0))


def _position_metric_series(
    metrics: dict[str, float], prefix: str, name: str
) -> list[float]:
    values: list[float] = []
    pos = 0
    while True:
        key = f"{prefix}/{name}/{pos}"
        if key not in metrics:
            break
        values.append(float(metrics[key]))
        pos += 1
    return values


def _weighted_average(values: list[float], counts: list[float]) -> float | None:
    if not values or not counts:
        return None
    total_count = sum(counts[: len(values)])
    if total_count <= 0:
        return None
    return (
        sum(value * count for value, count in zip(values, counts, strict=False))
        / total_count
    )


def _simulated_accept_length(accuracies: list[float]) -> float:
    cumulative = 1.0
    simulated = 0.0
    for accuracy in accuracies:
        cumulative *= max(0.0, min(1.0, float(accuracy)))
        simulated += cumulative
    return simulated


def _standalone_step_metrics(
    trainer: DrafterBaseTrainer,
    *,
    successful_steps: int,
    attempted_batches: int,
    step_elapsed_sec: float,
) -> dict[str, float]:
    raw_metrics = trainer.get_training_metrics()
    metrics: dict[str, float] = {
        key: float(value) for key, value in raw_metrics.items()
    }
    prefix = _block_metric_prefix(trainer)
    if prefix is not None:
        anchor_offset = 1 if prefix == "dflash" else 0
        losses = _position_metric_series(raw_metrics, prefix, "loss_per_position")
        accuracies = _position_metric_series(
            raw_metrics, prefix, "accuracy_per_position"
        )
        counts = _position_metric_series(raw_metrics, prefix, "count_per_position")
        pred_losses = losses[anchor_offset:]
        pred_accuracies = accuracies[anchor_offset:]
        pred_counts = counts[anchor_offset:]

        avg_loss = _weighted_average(pred_losses, pred_counts)
        avg_acc = _weighted_average(pred_accuracies, pred_counts)
        if avg_loss is not None:
            metrics["train/avg_loss"] = avg_loss
        if avg_acc is not None:
            metrics["train/avg_acc"] = avg_acc
        if pred_accuracies:
            metrics["train/simulated_acc_len"] = _simulated_accept_length(
                pred_accuracies
            )
        if f"{prefix}/top1_acc" in raw_metrics:
            metrics["train/top1_acc"] = float(raw_metrics[f"{prefix}/top1_acc"])
        if f"{prefix}/top5_acc" in raw_metrics:
            metrics["train/top5_acc"] = float(raw_metrics[f"{prefix}/top5_acc"])
        for idx, value in enumerate(pred_losses):
            metrics[f"train/ploss_{idx}"] = float(value)
        for idx, value in enumerate(pred_accuracies):
            metrics[f"train/acc_{idx}"] = float(value)
    metrics["train/step"] = float(successful_steps)
    metrics["train/global_step"] = float(
        getattr(trainer, "training_steps", successful_steps)
    )
    metrics["train/lr"] = _current_learning_rate(trainer)
    metrics["drafter/train_successful_steps"] = float(successful_steps)
    metrics["drafter/train_attempted_batches"] = float(attempted_batches)
    metrics["perf/step_time"] = float(step_elapsed_sec)
    return metrics


def _log_standalone_step_metrics(metrics: dict[str, float], *, rank: int) -> None:
    if rank != 0:
        return
    fields = [f"step={int(metrics.get('train/step', 0.0))}"]
    for key, label in (
        ("train/avg_loss", "avg_loss"),
        ("train/avg_acc", "avg_acc"),
        ("train/top1_acc", "top1"),
        ("train/top5_acc", "top5"),
        ("train/simulated_acc_len", "sim_acc_len"),
        ("train/lr", "lr"),
        ("perf/step_time", "step_time"),
        ("replay/cache_hit_ratio", "cache_hit"),
        ("replay/target_forward_time_total", "target_forward_total"),
    ):
        if key not in metrics:
            continue
        value = float(metrics[key])
        if key == "train/lr":
            fields.append(f"{label}={value:.3e}")
        elif key in {"perf/step_time", "replay/target_forward_time_total"}:
            fields.append(f"{label}={value:.3f}s")
        else:
            fields.append(f"{label}={value:.4f}")
    logger.warning("[standalone drafter metrics] %s", " ".join(fields))


def _init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        _configure_device(local_rank)
        device_name = str(get_device_name()).lower()
        if device_name == "npu":
            backend = "hccl"
        elif device_name == "cuda":
            backend = "nccl"
        elif device_name == "cpu":
            backend = "gloo"
        else:
            raise ValueError(
                f"Unsupported standalone drafter device_name={device_name!r}"
            )
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _configure_device(local_rank: int) -> None:
    device_name = get_device_name()
    device_module = get_torch_device()
    if device_name == "cpu":
        return
    set_device = getattr(device_module, "set_device", None)
    if callable(set_device):
        set_device(int(local_rank))


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return bool(value)
    ready = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    return bool(ready.item())


def _sync_any_rank_saved_checkpoint(saved: Any) -> bool:
    if not dist.is_initialized():
        return bool(saved)
    device = torch.device(get_device_name())
    flag = torch.tensor([1 if saved else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def log_resolved_config(config) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.warning(
            "Resolved SPECO standalone draft trainer config:\n%s",
            OmegaConf.to_yaml(config),
        )
