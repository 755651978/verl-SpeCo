# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise review fixes through production methods without device dependencies."""

import ast
import asyncio
import logging
import random
import time
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
        method.decorator_list = [
            decorator
            for decorator in method.decorator_list
            if isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        ]
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


@pytest.mark.parametrize("local_flags", [(True, True), (True, False)])
def test_bubble_batch_readiness_uses_writer_group_not_missing_dp(local_flags):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2, timeout=3)
    tensors = []
    groups = []

    class Scalar:
        def __init__(self, values, **kwargs):
            self.value = values[0]

        def item(self):
            return self.value

    def reduce(tensor, *, op, group):
        tensors.append(tensor)
        groups.append(group)
        barrier.wait()
        value = min(item.value for item in tensors)
        barrier.wait()
        tensor.value = value

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
        time=time,
        torch=SimpleNamespace(tensor=Scalar, int32="int32"),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MIN="min")),
    )
    workers = []
    for _ in range(2):
        worker = cls()
        worker._bubble_time_enabled = True
        worker.runtime_device = "cpu"
        worker.record_training_timing = lambda name, elapsed: None
        worker._full_training_group = lambda: "writer-group"
        worker._full_training_world_size = lambda: 2
        # replica-local initialization has no DP group. It must not be used.
        worker._get_dp_group = lambda: None
        worker._get_dp_world_size = lambda: 1
        workers.append(worker)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(w._sync_batch_readiness, flag)
            for w, flag in zip(workers, local_flags, strict=True)
        ]
        results = [future.result(timeout=5) for future in futures]
    assert results == [all(local_flags)] * 2
    assert groups == ["writer-group", "writer-group"]


def test_sync_readiness_does_not_add_writer_collective():
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
    )
    worker = cls()
    worker._bubble_time_enabled = False
    worker._get_dp_group = lambda: None
    worker._get_dp_world_size = lambda: 1
    assert worker._sync_batch_readiness(True)
    assert not worker._sync_batch_readiness(False)


def test_sync_readiness_keeps_dp_group_and_no_bubble_metrics():
    calls = []

    class Scalar:
        def __init__(self, values, **kwargs):
            self.value = values[0]

        def item(self):
            return self.value

    def reduce(tensor, *, op, group):
        calls.append(group)
        tensor.value = 0

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_sync_batch_readiness"},
        torch=SimpleNamespace(tensor=Scalar, int32="int32"),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MIN="min")),
    )
    worker = cls()
    worker._bubble_time_enabled = False
    worker.runtime_device = "cpu"
    worker._get_dp_group = lambda: "sync-dp"
    worker._get_dp_world_size = lambda: 2
    worker._full_training_group = lambda: pytest.fail(
        "Sync must not use the Bubble writer group"
    )
    worker.record_training_timing = lambda *args: pytest.fail(
        "Sync must not record Bubble readiness timing"
    )
    assert not worker._sync_batch_readiness(True)
    assert calls == ["sync-dp"]


@pytest.mark.parametrize("ready", [True, False])
def test_sync_step_preserves_preparation_timing_and_sample_bounds(ready):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_training_step_impl"},
        time=time,
    )
    worker = cls()
    worker.model = object()
    worker._bubble_time_enabled = False
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(
                training={"collect_hidden_states_from_old_logprob": True}
            )
        )
    )
    events = []

    def prepare(**bounds):
        events.append(("prepare", bounds))
        return object()

    worker.prepare_training_batch = prepare
    worker._prepare_training_batch_for_step = lambda **kwargs: pytest.fail(
        "Sync must not enter the Bubble preparation helper"
    )
    worker.record_training_timing = lambda name, elapsed: events.append(
        ("timing", name)
    )
    worker._sync_batch_readiness = (
        lambda flag: events.append(("readiness", flag)) or ready
    )

    async def train(batch, step):
        events.append(("train", step))
        return True

    worker._training_step_on_batch = train
    worker._consume_last_training_batch = lambda: events.append(("consume", None))
    assert (
        asyncio.run(worker._training_step_impl(4, min_sample_step=2, max_sample_step=3))
        is ready
    )
    assert events[:3] == [
        ("prepare", {"min_sample_step": 2, "max_sample_step": 3}),
        ("timing", "timing_s/drafter_prepare_batch"),
        ("readiness", True),
    ]
    assert events[3:] == ([("train", 4), ("consume", None)] if ready else [])


@pytest.mark.parametrize("local_error", [False, True])
def test_bubble_preparation_failure_participates_in_readiness(local_error):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_prepare_training_batch_for_step"},
    )
    worker = cls()
    worker._bubble_time_enabled = True
    flags = []

    def prepare():
        if local_error:
            raise MemoryError("rank-local preparation failure")
        return object()

    worker.prepare_training_batch = prepare
    worker._sync_batch_readiness = lambda flag: flags.append(flag) or False
    assert worker._prepare_training_batch_for_step() is None
    assert flags == [not local_error]
    if local_error:
        assert worker.last_training_error["type"] == "MemoryError"
    else:
        assert getattr(worker, "last_training_error", None) is None


def _status_trainer(
    *, bubble=False, use_logits=True, model_type="dspark", use_buffer=True
):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"get_training_data_status", "_filter_training_data_by_step"},
        torch=SimpleNamespace(is_tensor=lambda value: False),
    )
    worker = cls()
    samples = [
        {"step": version, "target_version": version, "loss_tokens": 32}
        for version in (2, 3, 4)
        for _ in range(4)
    ]
    worker.data_buffer = SimpleNamespace(
        get_all_data=lambda: list(samples),
        get_available_data=lambda **kwargs: list(samples),
        get_data_from_last_n_steps=lambda n: [s for s in samples if s["step"] >= 4 - n],
    )
    worker._bubble_time_enabled = bubble
    worker._target_lm_head_snapshots = {2: object(), 3: object(), 4: object()}
    worker._target_lm_head_weight_step = 9
    worker.current_rl_step = 4
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"use_logits": use_logits})
        )
    )
    worker.backend = SimpleNamespace(model_type=model_type)
    worker.use_data_buffer = use_buffer
    worker.collected_data = list(samples)
    worker.batch_size = 4
    worker.buffer_version = 1
    return worker


@pytest.mark.parametrize("use_logits", [False, True])
def test_sync_status_preserves_full_cross_step_pool(use_logits):
    worker = _status_trainer(use_logits=use_logits)
    status = worker.get_training_data_status(
        sample_last_n_steps=2, require_full_batch=True
    )
    assert status["trainable_samples"] == 12
    assert status["trainable_batches"] == 3
    assert (status["min_sample_step"], status["max_sample_step"]) == (2, 4)
    assert status["target_version"] == 9


@pytest.mark.parametrize("use_buffer", [False, True])
def test_sync_status_honors_explicit_sample_bounds(use_buffer):
    worker = _status_trainer(use_buffer=use_buffer)
    status = worker.get_training_data_status(min_sample_step=3, max_sample_step=3)
    assert status["trainable_samples"] == 4
    assert status["data_version"] == 3
    assert (status["min_sample_step"], status["max_sample_step"]) == (3, 3)


@pytest.mark.parametrize("use_buffer", [False, True])
def test_sync_batch_sampling_honors_the_same_status_bounds(use_buffer):
    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"_prepare_training_batch", "_filter_training_data_by_step"},
        random=random,
    )
    worker = cls()
    status_worker = _status_trainer(use_buffer=use_buffer)
    worker.__dict__.update(status_worker.__dict__)
    data = status_worker.data_buffer

    class Buffer:
        def __len__(self):
            return len(data.get_all_data())

        def get_all_data(self):
            return data.get_all_data()

    worker.data_buffer = Buffer()
    worker._active_training_reservation_id = None
    worker.training_steps = 0
    selected = []

    class SelectionReached(Exception):
        pass

    def sample(items, batch_size, rng):
        selected.extend(items)
        raise SelectionReached()

    worker._sample_training_items = sample
    with pytest.raises(SelectionReached):
        worker._prepare_training_batch(min_sample_step=3, max_sample_step=3)
    assert len(selected) == 4
    assert {sample["step"] for sample in selected} == {3}


def test_sync_eagle_without_logits_keeps_current_step_only():
    worker = _status_trainer(use_logits=False, model_type="eagle3")
    status = worker.get_training_data_status(sample_last_n_steps=2)
    assert status["trainable_samples"] == 4
    assert status["min_sample_step"] == 4


@pytest.mark.parametrize("target_version, expected", [(None, 4), (3, 3)])
def test_bubble_status_keeps_fixed_single_version(target_version, expected):
    worker = _status_trainer(bubble=True)
    # Explicit Sync bounds must not override a retained Bubble version.
    status = worker.get_training_data_status(
        target_version=target_version, min_sample_step=2, max_sample_step=2
    )
    assert status["trainable_samples"] == 4
    assert status["target_version"] == expected
    assert status["min_sample_step"] == expected


@pytest.mark.parametrize("accumulation_steps", [2, 4, 6])
def test_streamed_accumulation_keeps_one_batch_and_sampling_order(accumulation_steps):
    import weakref
    from contextlib import nullcontext

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {
            "training_accumulation_step",
            "_prepare_training_batch_for_step",
            "_sample_replay_items",
            "_sample_training_items",
        },
        torch=SimpleNamespace(enable_grad=nullcontext),
        random=random,
    )
    worker = cls()
    sampling = _sampling_trainer()
    worker.__dict__.update(sampling.__dict__)
    worker._bubble_time_enabled = True
    worker._sampling_accumulation_steps = accumulation_steps
    pool = [{"id": i, "_verl_is_hard": i < 5} for i in range(accumulation_steps * 4)]
    expected = []
    for index in range(accumulation_steps):
        sampling._sampling_accumulation_steps = accumulation_steps
        sampling._sampling_micro_index = index
        available = [
            s for s in pool if s["id"] not in {v for batch in expected for v in batch}
        ]
        expected.append([s["id"] for s in sampling._sample_replay_items(available, 2)])

    class Batch:
        pass

    live = weakref.WeakSet()
    sequence = []
    zeros = []
    consumed = []

    def prepare():
        assert len(live) == 0, (
            "previous device batch must be released before preparing the next"
        )
        available = [
            s for s in pool if id(s) not in worker._active_accumulation_item_ids
        ]
        worker._last_prepared_training_items = worker._sample_replay_items(available, 2)
        batch = Batch()
        batch.samples = list(worker._last_prepared_training_items)
        live.add(batch)
        return batch

    async def forward(batch, step, *, accumulation_steps, accumulation_index):
        assert len(live) == 1
        sequence.append([s["id"] for s in batch.samples])
        random.random()  # Forward RNG must not perturb deterministic replay sampling.
        if accumulation_index == accumulation_steps - 1:
            worker.optimizer_steps_total += 1
        return True

    worker.prepare_training_batch = prepare
    worker._sync_batch_readiness = lambda ready: ready
    worker.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: zeros.append(1))
    worker._training_step_on_batch = forward
    worker._consume_training_items = lambda items: consumed.extend(items)
    assert asyncio.run(worker.training_accumulation_step(4, accumulation_steps))
    assert sequence == expected
    assert worker.optimizer_steps_total == 1
    assert len(consumed) == 2 * accumulation_steps
    assert len({id(s) for s in consumed}) == len(consumed)
    assert len(live) == 0
    assert worker._active_accumulation_item_ids == set()
    assert zeros == [1]


def test_streamed_accumulation_discards_partial_gradients_on_peer_failure():
    from contextlib import nullcontext

    cls = _methods(
        "verl_speco/trainer/base_trainer.py",
        "DrafterBaseTrainer",
        {"training_accumulation_step", "_prepare_training_batch_for_step"},
        torch=SimpleNamespace(enable_grad=nullcontext),
    )
    worker = cls()
    worker._bubble_time_enabled = True
    worker.last_training_error = {"type": "StaleError", "message": "prior plan"}
    worker._last_prepared_training_items = [{"id": 1}]
    worker.prepare_training_batch = lambda: object()
    ready_calls = []
    worker._sync_batch_readiness = (
        lambda flag: ready_calls.append(flag) or len(ready_calls) == 1
    )
    gradients = []
    worker.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: gradients.clear())

    async def forward(*args, **kwargs):
        gradients.append(1)
        return True

    worker._training_step_on_batch = forward
    worker._consume_training_items = lambda items: pytest.fail(
        "failed step must not consume samples"
    )
    assert not asyncio.run(worker.training_accumulation_step(4, 2))
    assert len(ready_calls) == 2
    assert gradients == []
    assert worker._active_accumulation_item_ids == set()
    assert worker.last_training_error is None


@pytest.mark.parametrize("mode", ["poll", "wait"])
def test_bubble_async_completion_preserves_main_ci_evidence(mode):
    from verl_speco.integration.ci_result import CIResult

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_poll_pending_drafter_training",
            "_speco_wait_pending_drafter_training",
        },
    )
    driver = cls()
    plan = SimpleNamespace(
        source_global_step=4,
        plan_id="quota-4",
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
    )
    outcome = SimpleNamespace(
        trained=True,
        successful_steps=2,
        elapsed_sec=0.5,
        reason="trained",
        raw_results=[{"errors": [{"type": "TestError", "message": "test evidence"}]}],
        metrics={"drafter/trained": 1, "drafter/train_successful_steps_max": 2},
    )
    results = [outcome, None]

    def complete(**kwargs):
        return results.pop(0)

    driver._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        poll_pending_training=complete, wait_pending_training=complete
    )
    driver._speco_get_drafter_runtime_state = lambda: SimpleNamespace(active_plan=plan)
    driver._speco_ci_result = CIResult(training=True)
    released = []
    driver._speco_release_rollout_training_lease = lambda p: released.append(p.plan_id)
    method = (
        driver._speco_poll_pending_drafter_training
        if mode == "poll"
        else driver._speco_wait_pending_drafter_training
    )
    assert method() == (plan, outcome)
    assert method() == (plan, None)
    assert released == ["quota-4"]
    assert driver._speco_ci_result.data["optimizer_steps"] == 2
    assert driver._speco_ci_result.data["errors"] == outcome.raw_results[0]["errors"]


def _freeze_publish_driver(*, bubble=True, asynchronous=True, failure=None):
    from typing import Any, cast

    from verl_speco.trainer.drafter_freeze_policy import DrafterFreezePolicy
    from verl_speco.trainer.scheduler import AfterWeightUpdateContext

    cls = _methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        {
            "_speco_publish_drafter_weights",
            "_speco_update_rollout_drafter_weights",
            "_speco_wait_pending_drafter_publish_rpc",
            "_speco_validate_bubble_publish_acks",
            "_speco_freeze_observe_publish",
        },
        time=time,
        cast=cast,
        Any=Any,
        AfterWeightUpdateContext=AfterWeightUpdateContext,
        DrafterExecutionStrategy=DrafterExecutionStrategy,
    )
    driver = cls()
    driver.global_steps = 4
    driver._speco_freeze_policy = DrafterFreezePolicy({"enabled": True})
    driver._speco_rollout_idle_worker_enabled = lambda: bubble
    driver._speco_drafter_schedule_config = DrafterScheduleConfig
    driver._speco_freeze_publish_cost_seconds = lambda metrics: None
    driver._pending_drafter_publish_refs = None
    driver._pending_drafter_publish_contexts = []
    driver._speco_record_writer_switch_decision = lambda: None
    driver._ray_get_if_needed = lambda value: value
    driver.actor_rollout_wg = SimpleNamespace(world_size=2)
    stages = []
    applied = []
    completed = []

    def acks(version, *, staged=False):
        records = [
            {
                "worker_id": str(rank),
                "staged": staged,
                "published": not staged,
                "staged_version" if staged else "published_version": version,
            }
            for rank in range(2)
        ]
        if staged and failure == "stage":
            records[1]["staged_version"] = version - 1
        if not staged and failure == "apply":
            records[1]["published"] = False
        return records

    def rpc(method):
        def update(payload, *, global_steps):
            if failure == "submit":
                raise RuntimeError("submit failed")
            staged = method == "stage_draft_weights_async"
            stages.append(method)
            return acks(global_steps, staged=staged)

        return update

    def commit(version):
        if failure == "commit_rpc":
            raise RuntimeError("commit RPC failed")
        applied.append(version)
        return acks(version)

    driver._speco_actor_rollout_method = rpc
    driver.speco_commit_staged_drafter_weights = commit
    scheduler = SimpleNamespace(
        _publish_executor=object(),
        _flatten_publish_acknowledgements=DrafterScheduler._flatten_publish_acknowledgements,
        record_rollout_drafter_publish_completed=lambda version,
        **kwargs: completed.append(version),
        record_training_quota_publish_completed=lambda step: None,
    )

    def publish(context):
        driver._speco_update_rollout_drafter_weights(
            {}, context.global_step, asynchronous
        )
        return SimpleNamespace(
            publish_plan=SimpleNamespace(
                asynchronous=asynchronous, source_global_step=context.global_step
            ),
            publish_outcome=SimpleNamespace(published=True),
            metrics={"drafter/published": 1},
        )

    scheduler.on_safe_point = publish
    driver._speco_get_drafter_scheduler = lambda: scheduler
    plan = SimpleNamespace(
        execution_strategy=(
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            if bubble
            else DrafterExecutionStrategy.SYNC
        ),
        plan_id="quota-1",
        target_worker_ids=("0", "1"),
    )
    return driver, plan, stages, applied, completed


def test_bubble_freeze_advances_once_only_after_valid_apply_ack():
    driver, plan, stages, applied, completed = _freeze_publish_driver()
    driver._speco_publish_drafter_weights(True, plan)
    assert stages == ["stage_draft_weights_async"]
    assert driver._speco_freeze_policy.drafter_version == 0
    assert applied == [] and completed == []
    assert driver._speco_wait_pending_drafter_publish_rpc() == 2
    assert applied == [4] and completed == [4]
    assert driver._speco_freeze_policy.drafter_version == 1
    assert driver._speco_wait_pending_drafter_publish_rpc() == 0
    assert driver._speco_freeze_policy.drafter_version == 1


@pytest.mark.parametrize("failure", ["submit", "stage", "apply", "commit_rpc"])
def test_bubble_freeze_does_not_advance_on_publish_failure(failure):
    driver, plan, _, _, completed = _freeze_publish_driver(failure=failure)
    with pytest.raises(RuntimeError):
        driver._speco_publish_drafter_weights(True, plan)
        assert driver._speco_freeze_policy.drafter_version == 0
        driver._speco_wait_pending_drafter_publish_rpc()
    assert driver._speco_freeze_policy.drafter_version == 0
    assert completed == []


def test_bubble_direct_publish_advances_freeze_after_valid_apply():
    driver, plan, stages, applied, completed = _freeze_publish_driver(
        asynchronous=False
    )
    driver._speco_publish_drafter_weights(True, plan)
    assert stages == ["update_draft_weights"]
    assert completed == [4]
    assert applied == []
    assert driver._speco_freeze_policy.drafter_version == 1


def test_bubble_failed_direct_apply_does_not_advance_freeze():
    driver, plan, _, _, completed = _freeze_publish_driver(
        asynchronous=False, failure="apply"
    )
    with pytest.raises(RuntimeError, match="Invalid Bubble publish ACK"):
        driver._speco_publish_drafter_weights(True, plan)
    assert driver._speco_freeze_policy.drafter_version == 0
    assert completed == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sync_publish_keeps_original_outer_freeze_observer(asynchronous):
    driver, plan, stages, applied, completed = _freeze_publish_driver(
        bubble=False, asynchronous=asynchronous
    )
    metrics = driver._speco_publish_drafter_weights(True, plan)
    if asynchronous:
        driver._speco_wait_pending_drafter_publish_rpc()
    assert driver._speco_freeze_policy.drafter_version == 0
    assert stages == [
        "update_draft_weights_async" if asynchronous else "update_draft_weights"
    ]
    assert applied == [] and completed == [4]
    # The original Sync fit hook remains responsible for its one notification.
    driver._speco_freeze_observe_publish(metrics)
    assert driver._speco_freeze_policy.drafter_version == 1
