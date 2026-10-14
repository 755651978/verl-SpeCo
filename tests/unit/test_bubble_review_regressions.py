# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise review fixes through production methods without device dependencies."""

import ast
import asyncio
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl_speco.trainer.scheduler import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    TrainingDataStatus,
)


def _methods(path, class_name, names, **namespace):
    path = Path(__file__).resolve().parents[2] / path
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [
        n
        for n in original.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    for method in methods:
        method.decorator_list = []
    body = ast.ClassDef(
        name="Subject", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            body,
        ],
        type_ignores=[],
    )
    namespace.setdefault("logger", logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Subject"]


def _sampling_trainer():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sample_training_items", "_sample_replay_items"},
        random=random,
    )
    obj = cls()
    obj._active_training_target_version = 4
    obj._active_training_replay_cursor = 0
    obj.optimizer_steps_total = 0
    obj._sampling_accumulation_steps = 2
    obj._sampling_micro_index = 0
    obj._is_block_drafter_backend = lambda: True
    obj._block_drafter_metric_prefix = lambda: "dspark"
    obj._block_drafter_config_value = (
        lambda key, default: 0.1 if key == "hard_sample_ratio" else default
    )
    return obj


def _draw_micro_batch(obj, pool):
    items = obj._sample_replay_items(pool, 2)
    if obj._sampling_micro_index == 1:
        obj.optimizer_steps_total += 1
    obj._sampling_micro_index = 1 - obj._sampling_micro_index
    return items


def test_quota_replay_keeps_hard_sample_fraction_across_accumulated_steps():
    obj = _sampling_trainer()
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    batches = [_draw_micro_batch(obj, pool) for _ in range(40)]
    assert sum(item["_verl_is_hard"] for batch in batches for item in batch) == 8
    assert obj._active_training_replay_cursor == 80
    assert len(pool) == 20


def test_replay_sampling_continues_identically_after_short_plan_handoff():
    uninterrupted = _sampling_trainer()
    split = _sampling_trainer()
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    expected = [_draw_micro_batch(uninterrupted, pool) for _ in range(12)]
    actual = [_draw_micro_batch(split, pool) for _ in range(6)]
    resumed = _sampling_trainer()
    resumed._active_training_replay_cursor = split._active_training_replay_cursor
    resumed.optimizer_steps_total = split.optimizer_steps_total
    actual.extend(_draw_micro_batch(resumed, pool) for _ in range(6))
    assert actual == expected


def test_hard_fractional_quota_does_not_restart_rounding_each_cycle():
    obj = _sampling_trainer()
    obj._block_drafter_config_value = (
        lambda key, default: 0.03 if key == "hard_sample_ratio" else default
    )
    pool = [{"id": i, "_verl_is_hard": i < 10} for i in range(20)]
    hard = 0
    for _ in range(3):
        obj._active_training_replay_cursor = 0
        for _ in range(40):
            hard += sum(item["_verl_is_hard"] for item in _draw_micro_batch(obj, pool))
    assert hard == 7  # round(3 cycles * 20 optimizer steps * 4 samples * .03)


@pytest.mark.parametrize("bubble,fsdp", [(True, True), (False, True), (True, False)])
def test_fsdp1_bubble_uses_full_model_norm_and_sync_retains_its_path(bubble, fsdp):
    calls = []

    class FakeFSDP:
        def parameters(self):
            return ["local shard"]

        def clip_grad_norm_(self, *, max_norm):
            calls.append(("full_norm", max_norm))
            return 5.0

    def local_clip(parameters, *, max_norm):
        calls.append(("local_norm", max_norm))
        return 4.0

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_clip_training_grad_norm"},
        FSDP=FakeFSDP,
        torch=SimpleNamespace(
            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=local_clip))
        ),
    )
    obj = cls()
    obj._bubble_time_enabled = bubble
    obj.model = (
        FakeFSDP() if fsdp else SimpleNamespace(parameters=lambda: ["parameter"])
    )
    expected = "full_norm" if bubble and fsdp else "local_norm"
    assert obj._clip_training_grad_norm() == (5.0 if expected == "full_norm" else 4.0)
    assert calls == [(expected, 1.0)]


@pytest.mark.parametrize("fail", [False, True])
def test_cold_bubble_cleanup_requires_parking_and_preserves_training_progress(fail):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py", "DrafterBaseTrainer", {"cleanup_training"}
    )
    obj = cls()
    obj._bubble_time_enabled = True
    obj._pending_checkpoint_future = None
    obj._pending_full_checkpoint_future = None
    obj._full_checkpoint_executor = None
    obj.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: None)
    obj.training_steps = 17
    obj.optimizer_steps_total = 17
    calls = []

    async def park(*, strict):
        calls.append(strict)
        if fail:
            raise RuntimeError("model offload failed")

    obj.release_training_memory_after_activation = park
    if fail:
        with pytest.raises(RuntimeError, match="offload failed"):
            asyncio.run(obj.cleanup_training(clear_data=False, keep_hot=False))
    else:
        asyncio.run(obj.cleanup_training(clear_data=False, keep_hot=False))
    assert calls == [True]
    assert obj.training_steps == obj.optimizer_steps_total == 17


def test_actor_timing_and_feedback_source_are_idempotent_across_publish_merges():
    key = "_speco_drafter_timing_deducted_from_update_actor"
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_update_output_metrics"},
        _speco_metric_float=lambda v: None if v is None else float(v),
        _SPECO_DRAFTER_TIMING_DEDUCTED_KEY=key,
    )
    obj = cls()
    obj.global_steps = 7
    feedback_steps = []
    obj._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        record_step_metrics=lambda metrics,
        config,
        *,
        global_step: feedback_steps.append(global_step) or {}
    )
    obj._speco_drafter_schedule_config = lambda: DrafterScheduleConfig()
    output = SimpleNamespace(
        meta_info={
            "metrics": {
                "timing_s/update_actor": 10,
                "timing_s/drafter": 2,
                "timing_per_token_ms/update_actor": 1,
            }
        }
    )
    obj._speco_update_output_metrics(output, {"drafter/trained": 1})
    obj.global_steps = 8
    obj._speco_update_output_metrics(output, {"drafter/published": 1})
    assert output.meta_info["metrics"]["timing_s/update_actor"] == 8
    assert output.meta_info["metrics"]["timing_per_token_ms/update_actor"] == 0.8
    assert feedback_steps == [7, 7]


def test_one_rollout_cannot_trigger_slowdown_patience_twice():
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_gen_slowdown_patience=2,
    )
    scheduler.record_step_metrics(
        {"timing_per_token_ms/gen": 1.0}, config, global_step=1
    )
    slow = {"timing_per_token_ms/gen": 2.0, "drafter/idle_trained": 1}
    scheduler.record_step_metrics(slow, config, global_step=2)
    cap = scheduler._effective_idle_dynamic_batch_cap(config)
    scheduler.record_step_metrics(slow, config, global_step=2)
    assert scheduler._idle_worker_gen_slowdown_streak == 1
    assert scheduler._effective_idle_dynamic_batch_cap(config) == cap
    scheduler.record_step_metrics(slow, config, global_step=3)
    assert scheduler._idle_worker_gen_slowdown_streak == 2


@pytest.mark.parametrize("buffered", [False, True])
def test_sync_requires_successful_current_oldlogprob_collection(buffered):
    context = DrafterScheduleContext(
        global_step=4,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=True,
        data_status=TrainingDataStatus(
            current_step=4,
            current_step_samples=0,
            buffer_samples=8,
            trainable_samples=8,
            trainable_batches=4,
            batch_size_per_gpu=2,
            partial_batch_available=False,
            oldest_sample_step=2,
            newest_sample_step=2,
            same_step_data_required=False,
        ),
    )
    plan = DrafterScheduler().plan_training(
        context,
        DrafterScheduleConfig(training_interval_steps=4, use_data_buffer=buffered),
    )
    assert not plan.launch
    assert plan.reason == "no_current_step_oldlogprob_samples"


@pytest.mark.parametrize("group", [("0",), ("0", "2"), ("0", "1", "2", "3")])
def test_explicit_idle_groups_cannot_split_or_combine_runtime_collectives(group):
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    config = DrafterScheduleConfig(idle_worker_training_groups=(group,))
    with pytest.raises(ValueError, match="complete runtime training collectives"):
        scheduler._idle_training_groups(config)


def test_explicit_runtime_group_is_canonicalized_without_changing_topology():
    scheduler = DrafterScheduler()
    scheduler._metadata_idle_training_groups = (("0", "1"), ("2", "3"))
    assert scheduler._idle_training_groups(
        DrafterScheduleConfig(idle_worker_training_groups=(("3", "2"),))
    ) == (("2", "3"),)


@pytest.mark.parametrize("bubble", [True, False])
def test_target_head_version_cache_is_bubble_only(bubble):
    class Tensor:
        shape = (3, 2)

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"sync_target_lm_head_weight"},
        torch=SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor)),
    )
    obj = cls()
    obj._bubble_time_enabled = bubble
    obj._target_lm_head_snapshots = {}
    obj._target_lm_head_snapshot_limit = 2
    for step in range(3):
        obj.sync_target_lm_head_weight(
            Tensor(), global_step=step, defer_device_apply=True
        )
    assert set(obj._target_lm_head_snapshots) == ({1, 2} if bubble else set())


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "stale", "phase", "empty"]
)
def test_bubble_publish_requires_complete_same_version_acknowledgements(
    staged, failure
):
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_validate_bubble_publish_acks"},
    )
    obj = cls()
    obj.actor_rollout_wg = SimpleNamespace(world_size=4)
    obj._speco_get_drafter_scheduler = lambda: DrafterScheduler()
    version_key = "staged_version" if staged else "published_version"
    acks = [
        {"worker_rank": rank, version_key: 4, "staged": staged, "published": not staged}
        for rank in range(4)
    ]
    if failure == "missing":
        acks.pop()
    elif failure == "duplicate":
        acks[-1]["worker_rank"] = 0
    elif failure == "stale":
        acks[-1][version_key] = 3
    elif failure == "phase":
        acks[-1]["staged"] = not staged
        acks[-1]["published"] = staged
    elif failure == "empty":
        acks = []
    if failure:
        with pytest.raises(RuntimeError, match="Bubble publish ACK"):
            obj._speco_validate_bubble_publish_acks(acks, 4, staged=staged)
    else:
        obj._speco_validate_bubble_publish_acks(acks, 4, staged=staged)


def test_metadata_discovery_failure_cannot_fall_back_to_guessed_bubble_groups():
    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {"_speco_register_drafter_training_resource_metadata"},
    )
    obj = cls()
    obj.drafter_wg = object()
    obj._speco_rollout_idle_worker_enabled = lambda: True

    def fail_metadata():
        raise RuntimeError("worker metadata unavailable")

    obj.speco_get_drafter_training_resource_metadata = fail_metadata
    with pytest.raises(
        RuntimeError, match="Unsafe or unavailable Bubble training resource metadata"
    ):
        obj._speco_register_drafter_training_resource_metadata()


@pytest.mark.parametrize("module_name", ["sglang_runtime", "vllm_runtime"])
def test_runtime_recognizes_legacy_bubble_strategy(module_name):
    from importlib import import_module

    module = import_module(f"verl_speco.integration.{module_name}")
    config = {
        "training": {
            "execution_strategy": " ROLLOUT_IDLE_WORKER ",
            "scheduler": {"idle_worker": {"event_bus_name": "bus"}},
        }
    }
    assert module._rollout_idle_worker_config(config)["event_bus_name"] == "bus"


def test_inconsistent_updated_writer_shards_cannot_be_retried_as_zero_progress():
    from verl_speco.trainer.scheduler import TrainingOutcome, TrainingPlan

    scheduler = DrafterScheduler()
    scheduler._training_quota_debt_steps = 20
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=4,
        max_batches=2,
        publish_after_success=False,
    )
    outcome = TrainingOutcome(
        trained=False,
        successful_steps=1,
        worker_results=[],
        raw_results=[],
        elapsed_sec=1,
        reason="worker_result_inconsistent",
        metrics={},
    )
    with pytest.raises(RuntimeError, match="Inconsistent Bubble training state"):
        scheduler._record_training_outcome(plan, outcome)
    assert scheduler._training_quota_debt_steps == 20
    assert scheduler._idle_optimizer_steps_total == 0


def test_main_sample_bounds_survive_status_aggregation_and_plan_payload():
    from verl_speco.trainer.scheduler import ConservativeTrainingDataStatusPolicy

    values = {
        "current_step": 4,
        "current_step_samples": 2,
        "buffer_samples": 4,
        "trainable_samples": 4,
        "trainable_batches": 2,
        "batch_size_per_gpu": 2,
        "partial_batch_available": False,
        "oldest_sample_step": 2,
        "newest_sample_step": 4,
        "same_step_data_required": False,
        "min_sample_step": 2,
        "max_sample_step": 4,
        "worker_id": "0",
    }
    status = TrainingDataStatus.from_mapping(values)
    aggregate = ConservativeTrainingDataStatusPolicy().aggregate(
        [status], global_step=4
    )
    assert aggregate.min_sample_step == 2 and aggregate.max_sample_step == 4
    context = DrafterScheduleContext(
        global_step=4,
        training_mode="online",
        collected_samples_this_step=2,
        oldlogprob_collection_requested=False,
        data_status=aggregate,
    )
    plan = DrafterScheduler().plan_training(
        context, DrafterScheduleConfig(training_interval_steps=4)
    )
    assert plan.to_worker_payload()["min_sample_step"] == 2
    assert plan.to_worker_payload()["max_sample_step"] == 4
