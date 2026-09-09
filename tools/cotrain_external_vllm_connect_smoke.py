#!/usr/bin/env python3
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
"""Run the real co-train worker initialization, then stop before ``fit``.

Use the same Hydra overrides as ``python -m verl_speco.main``.  This entrypoint
intentionally reuses SpecoTaskRunner and SpecoRayPPOTrainer: dataset creation,
resource pools, WorkerDict construction, actor/rollout/ref model wrapping,
SpecoWorker construction, and external-vLLM initialization therefore follow
the production path.  Only the final PPO training loop is replaced by a
diagnostic success marker.
"""

from __future__ import annotations

import json
import inspect
import logging
import os
import socket
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import hydra
from omegaconf import OmegaConf

# Prefer the checkout under test to an older pip-installed verl-speco package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

from verl_speco.integration.task_runner import SpecoTaskRunner
from verl.single_controller.base.decorator import Dispatch, register
from verl_speco.integration.rollout_publish import DraftWeightPublishMixin


_PHASE_ENV = "SPECO_CONNECT_SMOKE_PHASE"
_EMPTY_ACTOR_PROFILER_ENV = "SPECO_CONNECT_SMOKE_EMPTY_ACTOR_PROFILER"
_INSTALL_ROLLOUT_RUNTIME_ENV = "SPECO_CONNECT_SMOKE_INSTALL_ROLLOUT_RUNTIME"
_INSTALL_OLDLOGPROB_RUNTIME_ENV = "SPECO_CONNECT_SMOKE_INSTALL_OLDLOGPROB_RUNTIME"
_SET_DEVICE_BEFORE_HCCL_ENV = "SPECO_CONNECT_SMOKE_SET_DEVICE_BEFORE_HCCL"
_PEER_WAIT_MODE_ENV = "SPECO_CONNECT_SMOKE_PEER_WAIT_MODE"
_MARKER_ENV = "SPECO_CONNECT_SMOKE_MARKER"
_BEFORE_INIT_MODEL = "before_init_model"
_AFTER_ACTOR_MODEL = "after_actor_model"
_AFTER_ACTOR_MODEL_NO_REF = "after_actor_model_no_ref"


def _mapping_value(value, key, default=None):
    if value is None:
        return default
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def _pre_model_weight_sync_config(worker) -> dict:
    rollout = _mapping_value(worker.config, "rollout")
    drafter = _mapping_value(rollout, "drafter")
    if drafter is None:
        serialized = getattr(worker, "_speco_sglang_drafter_config_env", "")
        drafter = json.loads(serialized) if serialized else None
    training = _mapping_value(drafter, "training")
    source = _mapping_value(training, "vllm_feature_source")
    hot_update = _mapping_value(source, "weight_hot_update")
    config = (
        OmegaConf.to_container(hot_update, resolve=True)
        if OmegaConf.is_config(hot_update)
        else dict(hot_update or {})
    )
    if not isinstance(config, dict) or not config.get("enabled", False):
        raise RuntimeError(
            "before_init_model requires "
            "vllm_feature_source.weight_hot_update.enabled=true"
        )
    endpoints = _mapping_value(source, "endpoints")
    config["endpoints"] = list(endpoints or ())
    if not config["endpoints"]:
        raise RuntimeError("before_init_model requires at least one vLLM endpoint")
    return config


def _before_init_model_probe(worker, *args, **kwargs):
    """Initialize verl's process group, then connect before building models."""
    import torch.distributed as dist

    from verl.utils.device import get_device_name, get_torch_device
    from verl.utils.distributed import initialize_global_process_group_ray
    from verl_speco.integration.rollout_publish import (
        install_oldlogprob_hidden_runtime_for_worker,
        install_rollout_runtime_for_worker,
    )
    from verl_speco.integration.external_vllm_weight_sync import (
        initialize_worker_weight_sync,
    )

    marker = Path(os.environ[_MARKER_ENV])
    rollout = _mapping_value(worker.config, "rollout")
    drafter = _mapping_value(rollout, "drafter")
    if drafter is None:
        serialized = getattr(worker, "_speco_sglang_drafter_config_env", "")
        drafter = json.loads(serialized) if serialized else None
    training = _mapping_value(drafter, "training")
    source = _mapping_value(training, "vllm_feature_source")
    hot_update = _mapping_value(source, "weight_hot_update", {})
    timeout = float(
        _mapping_value(hot_update, "timeout_seconds", 600)
    )
    rank = int(getattr(worker, "rank", -1))
    device_module = get_torch_device()
    if os.environ.get(_INSTALL_ROLLOUT_RUNTIME_ENV, "0") == "1":
        logger.warning(
            "[cotrain connect smoke] installing rollout runtime before "
            "external HCCL rank=%s",
            rank,
        )
        install_rollout_runtime_for_worker(worker)
    if os.environ.get(_INSTALL_OLDLOGPROB_RUNTIME_ENV, "0") == "1":
        logger.warning(
            "[cotrain connect smoke] installing old-logprob runtime before "
            "external HCCL rank=%s",
            rank,
        )
        install_oldlogprob_hidden_runtime_for_worker(worker)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    set_device_before_hccl = (
        os.environ.get(_SET_DEVICE_BEFORE_HCCL_ENV, "1") == "1"
    )
    if set_device_before_hccl:
        device_module.set_device(local_rank)
    logger.warning(
        "[cotrain connect smoke] before init_model enter rank=%s pid=%s "
        "device=%s set_device_before_hccl=%s local_rank=%s "
        "current_device=%s visible=%r "
        "dist_initialized=%s "
        "actor_exists=%s rollout_exists=%s",
        rank,
        os.getpid(),
        get_device_name(),
        set_device_before_hccl,
        local_rank,
        device_module.current_device(),
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        __import__("torch").distributed.is_initialized(),
        getattr(worker, "actor", None) is not None,
        getattr(worker, "rollout", None) is not None,
    )
    if not dist.is_initialized():
        logger.warning(
            "[cotrain connect smoke] before init_model initializing verl "
            "global process group rank=%s",
            rank,
        )
        initialize_global_process_group_ray(timeout_second=None)
    logger.warning(
        "[cotrain connect smoke] before init_model process group ready "
        "rank=%s dist_rank=%s dist_world_size=%s actor_exists=%s "
        "rollout_exists=%s",
        rank,
        dist.get_rank(),
        dist.get_world_size(),
        getattr(worker, "actor", None) is not None,
        getattr(worker, "rollout", None) is not None,
    )
    peer_wait_mode = os.environ.get(_PEER_WAIT_MODE_ENV, "marker").lower()
    if peer_wait_mode not in {"marker", "gloo"}:
        raise ValueError(
            f"{_PEER_WAIT_MODE_ENV} must be 'marker' or 'gloo', "
            f"got {peer_wait_mode!r}"
        )
    sync_config = _pre_model_weight_sync_config(worker)
    logger.warning(
        "[cotrain connect smoke] peer wait mode=%s rank=%s",
        peer_wait_mode,
        rank,
    )
    if peer_wait_mode == "gloo":
        control_group = dist.new_group(backend="gloo")
        caught = None
        local_error = None
        try:
            initialize_worker_weight_sync(worker, sync_config)
        except BaseException as exc:
            caught = exc
            local_error = f"rank={rank}: {type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        try:
            dist.all_gather_object(errors, local_error, group=control_group)
        finally:
            dist.destroy_process_group(control_group)
        if caught is not None:
            raise caught
        peer_errors = [error for error in errors if error is not None]
        if peer_errors:
            raise RuntimeError("; ".join(peer_errors))
        if rank == 0:
            marker.write_text("OK", encoding="utf-8")
        logger.warning(
            "[cotrain connect smoke] gloo peer wait completed rank=%s", rank
        )
        raise RuntimeError("EXPECTED_BEFORE_INIT_MODEL_CONNECT_PASS")

    if rank == 0:
        try:
            initialize_worker_weight_sync(worker, sync_config)
        except BaseException as exc:
            marker.write_text(f"ERROR:{type(exc).__name__}:{exc}", encoding="utf-8")
            raise
        marker.write_text("OK", encoding="utf-8")
        logger.warning(
            "[cotrain connect smoke] before init_model external HCCL connected"
        )
    else:
        deadline = time.monotonic() + timeout
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "rank 0 did not publish the before-init-model probe result"
                )
            time.sleep(0.1)
        result = marker.read_text(encoding="utf-8")
        if result != "OK":
            raise RuntimeError(f"rank 0 before-init-model probe failed: {result}")
        logger.warning(
            "[cotrain connect smoke] before init_model rank=%s observed success", rank
        )
    raise RuntimeError("EXPECTED_BEFORE_INIT_MODEL_CONNECT_PASS")


def _connect_from_actor_init_probe(worker, phase: str) -> None:
    """Connect rank 0 while peer ranks wait on a CPU/Gloo control group."""
    import torch.distributed as dist

    from verl_speco.integration.external_vllm_weight_sync import (
        initialize_worker_weight_sync,
    )

    marker = Path(os.environ[_MARKER_ENV])
    hot_update = _mapping_value(
        _mapping_value(
            _mapping_value(
                _mapping_value(worker.config, "rollout"), "drafter"
            ),
            "training",
        ),
        "vllm_feature_source",
    )
    hot_update = _mapping_value(hot_update, "weight_hot_update", {})
    timeout = float(_mapping_value(hot_update, "timeout_seconds", 600))
    rank = int(getattr(worker, "rank", -1))
    control_group = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=timeout + 120),
    )
    logger.warning(
        "[cotrain connect smoke] %s control group ready rank=%s backend=gloo",
        phase,
        rank,
    )
    error = None
    if rank == 0:
        try:
            initialize_worker_weight_sync(worker, _pre_model_weight_sync_config(worker))
        except BaseException as exc:
            error = f"{type(exc).__name__}:{exc}"
            marker.write_text(f"ERROR:{error}", encoding="utf-8")
        else:
            marker.write_text("OK", encoding="utf-8")
            logger.warning("[cotrain connect smoke] %s external HCCL connected", phase)
    errors = [None] * dist.get_world_size()
    try:
        dist.all_gather_object(errors, error, group=control_group)
    finally:
        dist.destroy_process_group(control_group)
    failures = [item for item in errors if item]
    if failures:
        raise RuntimeError(f"rank 0 {phase} probe failed: {failures[0]}")


def _after_actor_model_probe(worker, *args, **kwargs):
    """Stop immediately after the actor TrainingWorker finishes reset()."""
    phase = os.environ.get(_PHASE_ENV, _AFTER_ACTOR_MODEL)
    original_role = worker.role
    if phase == _AFTER_ACTOR_MODEL_NO_REF:
        if original_role != "actor_rollout_ref":
            raise RuntimeError(
                "after_actor_model_no_ref requires role='actor_rollout_ref', "
                f"got {original_role!r}"
            )
        worker.role = "actor_rollout"
        logger.warning(
            "[cotrain connect smoke] skipping ref model for diagnostic "
            "rank=%s original_role=%s probe_role=%s",
            getattr(worker, "rank", None),
            original_role,
            worker.role,
        )
    training_worker_cls = getattr(worker, "actor_worker_cls", None)
    if training_worker_cls is None:
        for owner_cls in type(worker).__mro__:
            candidate = owner_cls.__dict__.get("init_model")
            while candidate is not None:
                candidate_globals = getattr(candidate, "__globals__", {})
                training_worker_cls = candidate_globals.get("TrainingWorker")
                if training_worker_cls is not None:
                    break
                candidate = getattr(candidate, "__wrapped__", None)
            if training_worker_cls is not None:
                break
    if training_worker_cls is None or not hasattr(training_worker_cls, "reset"):
        raise RuntimeError(
            "Unable to locate the TrainingWorker.reset used by the installed "
            "ActorRolloutRefWorker.init_model"
        )
    logger.warning(
        "[cotrain connect smoke] located actor TrainingWorker class=%s.%s file=%s",
        training_worker_cls.__module__,
        training_worker_cls.__qualname__,
        inspect.getsourcefile(training_worker_cls),
    )
    original_reset = training_worker_cls.reset

    def reset_then_connect(training_worker, *reset_args, **reset_kwargs):
        result = original_reset(training_worker, *reset_args, **reset_kwargs)
        if getattr(worker, "actor", None) is not training_worker:
            return result
        engine = getattr(training_worker, "engine", None)
        logger.warning(
            "[cotrain connect smoke] after actor model rank=%s pid=%s "
            "engine=%s rollout_exists=%s",
            getattr(worker, "rank", None),
            os.getpid(),
            type(engine).__name__ if engine is not None else None,
            getattr(worker, "rollout", None) is not None,
        )
        _connect_from_actor_init_probe(worker, phase)
        raise RuntimeError(f"EXPECTED_{phase.upper()}_CONNECT_PASS")

    training_worker_cls.reset = reset_then_connect
    try:
        return super(_AfterActorModelDraftWeightPublishMixin, worker).init_model(
            *args, **kwargs
        )
    finally:
        training_worker_cls.reset = original_reset
        worker.role = original_role


class _BeforeInitModelDraftWeightPublishMixin(DraftWeightPublishMixin):
    """Diagnostic mixin serialized into the real ActorRolloutRef WorkerDict."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        empty_profiler = os.environ.get(_EMPTY_ACTOR_PROFILER_ENV, "1") == "1"
        actor_config = _mapping_value(config, "actor")
        profiler_config = _mapping_value(actor_config, "profiler")
        if not empty_profiler or actor_config is None or profiler_config is None:
            super().__init__(*args, **kwargs)
            return

        from omegaconf import OmegaConf, open_dict

        saved_profiler = OmegaConf.to_container(profiler_config, resolve=False)
        logger.warning(
            "[cotrain connect smoke] temporarily disabling actor profiler "
            "during Worker construction pid=%s",
            os.getpid(),
        )
        with open_dict(actor_config):
            actor_config.profiler = {}
        try:
            super().__init__(*args, **kwargs)
        finally:
            with open_dict(actor_config):
                actor_config.profiler = OmegaConf.create(saved_profiler)

    init_model = register(dispatch_mode=Dispatch.ONE_TO_ALL)(
        _before_init_model_probe
    )


class _AfterActorModelDraftWeightPublishMixin(DraftWeightPublishMixin):
    """Run the real init_model only through actor FSDP engine creation."""

    init_model = register(dispatch_mode=Dispatch.ONE_TO_ALL)(
        _after_actor_model_probe
    )


def _stop_before_fit(self) -> None:
    """Replacement for ``SpecoRayPPOTrainer.fit`` used only by this tool."""
    sync = getattr(self, "_speco_external_vllm_weight_sync", None)
    sender = None
    if sync is not None:
        sender = getattr(sync, "actor_worker_group", None)
    logger.warning(
        "[cotrain connect smoke] init_workers completed hostname=%s pid=%s "
        "external_sync_enabled=%s actor_worker_group=%s; skipping fit",
        socket.gethostname(),
        os.getpid(),
        bool(getattr(sync, "enabled", False)),
        type(sender).__name__ if sender is not None else None,
    )
    if not bool(getattr(sync, "enabled", False)):
        raise RuntimeError(
            "Real co-train initialization completed without enabling external "
            "vLLM weight sync; check collect_hidden_states_from_vllm and "
            "vllm_feature_source.weight_hot_update.enabled"
        )
    print("REAL_COTRAIN_EXTERNAL_VLLM_CONNECT_PASS", flush=True)


class ConnectOnlySpecoTaskRunner(SpecoTaskRunner):
    """Delegate to the production task runner with only ``fit`` patched out."""

    def run(self, config):
        from verl_speco.integration import rollout_publish
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        phase = os.environ.get(_PHASE_ENV, "after_init_workers")
        original_fit = SpecoRayPPOTrainer.fit
        original_init_workers = SpecoRayPPOTrainer.init_workers
        original_init_drafter = SpecoRayPPOTrainer._init_speco_drafter_workers
        original_init_producer = SpecoRayPPOTrainer._speco_init_vllm_feature_producer
        original_publish_mixin = rollout_publish.DraftWeightPublishMixin

        def logged_phase(label, operation):
            def wrapped(trainer, *args, **kwargs):
                logger.warning("[cotrain connect smoke] %s entering", label)
                try:
                    return operation(trainer, *args, **kwargs)
                finally:
                    logger.warning("[cotrain connect smoke] %s exited", label)

            return wrapped

        SpecoRayPPOTrainer.fit = _stop_before_fit
        SpecoRayPPOTrainer.init_workers = logged_phase(
            "production init_workers", original_init_workers
        )
        SpecoRayPPOTrainer._init_speco_drafter_workers = logged_phase(
            "SpecoWorker init", original_init_drafter
        )
        SpecoRayPPOTrainer._speco_init_vllm_feature_producer = logged_phase(
            "external vLLM producer/sender init", original_init_producer
        )
        if phase == _BEFORE_INIT_MODEL:
            rollout_publish.DraftWeightPublishMixin = (
                _BeforeInitModelDraftWeightPublishMixin
            )
        elif phase in {_AFTER_ACTOR_MODEL, _AFTER_ACTOR_MODEL_NO_REF}:
            rollout_publish.DraftWeightPublishMixin = (
                _AfterActorModelDraftWeightPublishMixin
            )
        logger.warning(
            "[cotrain connect smoke] entering production task runner "
            "hostname=%s pid=%s",
            socket.gethostname(),
            os.getpid(),
        )
        try:
            try:
                return super().run(config)
            except BaseException as exc:
                marker_path = os.environ.get(_MARKER_ENV)
                marker_ok = bool(
                    marker_path
                    and Path(marker_path).exists()
                    and Path(marker_path).read_text(encoding="utf-8") == "OK"
                )
                if (
                    phase
                    in {
                        _BEFORE_INIT_MODEL,
                        _AFTER_ACTOR_MODEL,
                        _AFTER_ACTOR_MODEL_NO_REF,
                    }
                    and marker_ok
                    and f"EXPECTED_{phase.upper()}_CONNECT_PASS" in str(exc)
                ):
                    print(
                        f"{phase.upper()}_EXTERNAL_VLLM_CONNECT_PASS",
                        flush=True,
                    )
                    return None
                raise
        finally:
            SpecoRayPPOTrainer.fit = original_fit
            SpecoRayPPOTrainer.init_workers = original_init_workers
            SpecoRayPPOTrainer._init_speco_drafter_workers = original_init_drafter
            SpecoRayPPOTrainer._speco_init_vllm_feature_producer = (
                original_init_producer
            )
            rollout_publish.DraftWeightPublishMixin = original_publish_mixin


@hydra.main(
    config_path="../verl_speco/config",
    config_name="speco_trainer",
    version_base=None,
)
def main(config) -> None:
    from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    from verl_speco.integration.compat import check_compatible_verl

    check_compatible_verl()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    logger.warning(
        "[cotrain connect smoke] temporary actor-profiler suppression=%s",
        os.environ.get(_EMPTY_ACTOR_PROFILER_ENV, "1"),
    )

    import ray

    phase = os.environ.get(_PHASE_ENV, "after_init_workers")
    if phase not in {
        "after_init_workers",
        _BEFORE_INIT_MODEL,
        _AFTER_ACTOR_MODEL,
        _AFTER_ACTOR_MODEL_NO_REF,
    }:
        raise ValueError(
            f"{_PHASE_ENV} must be 'after_init_workers', "
            f"'{_BEFORE_INIT_MODEL}', '{_AFTER_ACTOR_MODEL}', or "
            f"'{_AFTER_ACTOR_MODEL_NO_REF}'"
        )
    marker = Path(tempfile.gettempdir()) / (
        f"speco-connect-smoke-{socket.gethostname()}-{os.getpid()}.marker"
    )
    marker.unlink(missing_ok=True)
    os.environ[_MARKER_ENV] = str(marker)

    logger.warning(
        "[cotrain connect smoke] dispatching real SpecoTaskRunner phase=%s "
        "hostname=%s pid=%s marker=%s",
        phase,
        socket.gethostname(),
        os.getpid(),
        marker,
    )
    try:
        run_ppo(
            config,
            task_runner_class=ray.remote(num_cpus=1)(ConnectOnlySpecoTaskRunner),
        )
    finally:
        marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
