# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import time
from dataclasses import replace

import pytest

from verl_speco.trainer.scheduler import (
    CallbackDrafterWorkerExecutor,
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    TrainingDataStatus,
    TrainingOutcome,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.lifecycle import BeforeActorUpdateContext

try:
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer
except ModuleNotFoundError:
    SpecoRayPPOTrainer = None


class _FakeGenerationOutput:
    def __init__(self, samples=None):
        self.non_tensor_batch = {"drafter_sample": samples or []}
        self.meta_info = {"metrics": {}}


def _status(worker_id: str, *, batches: int = 5) -> TrainingDataStatus:
    return TrainingDataStatus(
        current_step=10,
        current_step_samples=batches,
        buffer_samples=batches,
        trainable_samples=batches,
        trainable_batches=batches,
        batch_size_per_gpu=1,
        partial_batch_available=False,
        oldest_sample_step=10,
        newest_sample_step=10,
        same_step_data_required=False,
        target_version=10,
        data_version=10,
        buffer_version=1,
        worker_id=worker_id,
        worker_incarnation=f"worker-{worker_id}",
    )


def _context() -> DrafterScheduleContext:
    return DrafterScheduleContext(
        global_step=10,
        training_mode="online",
        collected_samples_this_step=2,
        oldlogprob_collection_requested=False,
    )


def _idle_config() -> DrafterScheduleConfig:
    return DrafterScheduleConfig(
        training_interval_steps=5,
        train_batches_per_trigger=10,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        idle_worker_training_groups=(("0", "1"),),
        idle_worker_min_idle_window_sec=0.25,
        idle_worker_initial_batch_estimate_sec=0.9,
        idle_worker_deadline_guard_sec=0.2,
        idle_worker_max_batches_per_window=2,
    )


def _auto_idle_config(group_size: int) -> DrafterScheduleConfig:
    return replace(
        _idle_config(),
        idle_worker_training_groups=(),
        idle_worker_group_mode="auto",
        idle_worker_group_size=group_size,
    )


def _scheduler_with_statuses(worker_ids: tuple[str, ...]) -> DrafterScheduler:
    def inspect_data(
        sample_last_n_steps: int,
        require_full_batch: bool,
        requested_worker_ids: tuple[str, ...] | None = None,
    ):
        selected = requested_worker_ids or worker_ids
        return [
            {
                **_status(worker_id).__dict__,
                "available": True,
                "rank": int(worker_id) if str(worker_id).isdigit() else index,
            }
            for index, worker_id in enumerate(selected)
        ]

    return DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=inspect_data,
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )


def test_idle_worker_plan_requires_complete_training_group() -> None:
    scheduler = DrafterScheduler()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=103.0,
            event_ts=100.0,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), _idle_config())

    assert not plan.launch
    assert plan.reason == "incomplete_training_group"
    assert plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert plan.metrics()["bubble/skipped_incomplete_group"] == 1


def test_auto_idle_worker_groups_do_not_train_half_collective_group() -> None:
    scheduler = DrafterScheduler()
    deadline_ts = time.time() + 2.8
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="1",
            replica_rank=1,
        )
    )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), _auto_idle_config(2))

    assert not plan.launch
    assert plan.reason == "incomplete_training_group"


def test_auto_idle_worker_groups_select_complete_group_from_four_workers() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1", "2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=int(worker_id),
            )
        )
    for worker_id in ("2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )

    plan = scheduler.prepare_training_plan(_context(), _auto_idle_config(2))

    assert plan.launch
    assert plan.target_worker_ids == ("2", "3")
    assert plan.training_group_id == "idle-group-1"


def test_explicit_idle_worker_groups_override_auto_group_size() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1", "2", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=int(worker_id),
            )
        )
    for worker_id in ("0", "3"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(
            _auto_idle_config(2),
            idle_worker_training_groups=(("0", "3"),),
        ),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("0", "3")
    assert plan.training_group_id == "idle-group-0"


def test_idle_worker_auto_group_config_from_nested_mapping() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "scheduler": {
                "execution": {"strategy": "rollout_idle_worker"},
                "idle_worker": {"group_mode": "auto", "group_size": 2},
            }
        }
    )

    assert config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
    assert config.idle_worker_group_mode == "auto"
    assert config.idle_worker_group_size == 2
    assert config.idle_worker_training_groups == ()


def test_metadata_idle_worker_groups_wait_for_all_collective_replicas() -> None:
    scheduler = _scheduler_with_statuses(("0", "1", "2", "3"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1, 2, 3],
            },
            {
                "in_drafter_train_group": True,
                "replica_rank": 1,
                "training_group_ranks": [2, 3],
                "full_collective_ranks": [0, 1, 2, 3],
            },
        ]
    )
    config = replace(_auto_idle_config(2), idle_worker_group_size=None)
    deadline_ts = time.time() + 2.8
    for replica_rank in (0, 1):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=str(replica_rank),
                replica_rank=replica_rank,
            )
        )
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    half_plan = scheduler.prepare_training_plan(_context(), config)

    assert not half_plan.launch
    assert half_plan.reason == "incomplete_training_group"

    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="1",
            replica_rank=1,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.target_worker_ids == ("0", "1", "2", "3")
    assert plan.training_group_id == "idle-group-0"


def test_metadata_replica_groups_merge_multiple_ranks_for_same_replica() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0],
                "full_collective_ranks": [0, 1],
            },
            {
                "rank": 1,
                "worker_id": "1",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )
    deadline_ts = time.time() + 2.8
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.WORKER_IDLE,
            worker_id="0",
            replica_rank=0,
            memory_released=True,
            must_be_ready_at=deadline_ts,
        )
    )

    plan = scheduler.prepare_training_plan(
        _context(),
        replace(_auto_idle_config(2), idle_worker_group_size=None),
    )

    assert plan.launch
    assert plan.target_worker_ids == ("0", "1")


def test_metadata_replica_worker_mapping_is_available_for_fallback() -> None:
    scheduler = DrafterScheduler()
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "rank": 0,
                "worker_id": "0",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [0],
                "full_collective_ranks": [0, 1],
            },
            {
                "rank": 1,
                "worker_id": "1",
                "in_drafter_train_group": True,
                "replica_rank": 0,
                "training_group_ranks": [1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )

    assert scheduler.rollout_idle_replica_ranks() == (0,)
    assert scheduler.rollout_idle_worker_ids_for_replica(0) == ("0", "1")
    assert scheduler.rollout_idle_worker_ids_for_replica(
        1, fallback_worker_id="worker-1"
    ) == ("worker-1",)


def test_auto_idle_worker_without_metadata_or_group_size_fails_closed() -> None:
    scheduler = DrafterScheduler()
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )

    plan = scheduler.prepare_training_plan(
        _context(), replace(_auto_idle_config(1), idle_worker_group_size=None)
    )

    assert not plan.launch
    assert plan.reason == "missing_training_group_metadata"


def test_idle_worker_auto_budget_bootstraps_one_batch_without_manual_estimate() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    config = replace(
        _auto_idle_config(2),
        idle_worker_min_idle_window_sec=None,
        idle_worker_max_batches_per_window=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.max_batches == 1
    assert plan.reason == "training_ready"
    assert plan.idle_batch_estimate_sec == pytest.approx(0.25)


def test_idle_worker_budget_is_capped_by_observed_replica_idle_window() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                event_ts=100.0,
            )
        )
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.GENERATION_STARTED,
                worker_id=worker_id,
                replica_rank=replica_rank,
                event_ts=102.0,
            )
        )

    now = time.time()
    for replica_rank, worker_id in enumerate(("0", "1")):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=replica_rank,
                memory_released=True,
                must_be_ready_at=now + 30.0,
                event_ts=now,
            )
        )
    config = replace(
        _idle_config(),
        idle_worker_initial_batch_estimate_sec=0.5,
        idle_worker_deadline_guard_sec=0.1,
        idle_worker_max_batches_per_window=10,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.idle_window_sec == pytest.approx(2.0, abs=0.1)
    assert plan.max_batches == 3


def test_idle_worker_deadline_guard_learns_reclaim_cost() -> None:
    scheduler = DrafterScheduler()
    config = replace(_idle_config(), idle_worker_deadline_guard_sec=0.1)

    scheduler.record_reclaim_elapsed(0.7)

    assert scheduler._effective_idle_deadline_guard_sec(config) == pytest.approx(0.7)


def test_idle_worker_training_does_not_wait_for_training_interval() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    config = replace(
        _auto_idle_config(2),
        training_interval_steps=5,
        idle_worker_min_idle_window_sec=None,
        idle_worker_max_batches_per_window=None,
        idle_worker_initial_batch_estimate_sec=None,
        idle_worker_deadline_guard_sec=None,
    )
    context = DrafterScheduleContext(
        global_step=6,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
    )

    plan = scheduler.prepare_training_plan(context, config)

    assert not plan.interval_matched
    assert plan.launch
    assert plan.reason == "training_ready"
    assert plan.max_batches == 1


def test_idle_worker_plan_uses_buffered_data_target_version() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    deadline_ts = time.time() + 30.0
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
            )
        )
    buffered_status = replace(
        _status("group"),
        current_step=11,
        newest_sample_step=10,
        data_version=10,
        target_version=10,
        worker_snapshots={
            worker_id: {
                "buffer_version": 1,
                "data_version": 10,
                "worker_incarnation": f"worker-{worker_id}",
                "trainable_samples": 5,
            }
            for worker_id in ("0", "1")
        },
    )
    context = DrafterScheduleContext(
        global_step=11,
        training_mode="online",
        collected_samples_this_step=0,
        oldlogprob_collection_requested=False,
        data_status=buffered_status,
    )

    plan = scheduler.prepare_training_plan(context, _idle_config())

    assert plan.launch
    assert plan.source_global_step == 11
    assert plan.data_version == 10
    assert plan.required_target_version == 10


def test_idle_worker_publish_is_async_after_weight_update() -> None:
    scheduler = DrafterScheduler()
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
    )

    publish_plan = scheduler.plan_publish(
        global_step=10,
        drafter_trained=True,
        config=_idle_config(),
        training_plan=plan,
    )

    assert publish_plan.publish
    assert publish_plan.asynchronous


def test_idle_worker_starvation_guard_launches_sync_fallback_at_safe_point() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )
    config = replace(
        _auto_idle_config(1),
        idle_worker_group_size=None,
        idle_worker_fallback_to_sync=True,
        max_steps_without_training=2,
    )

    plan = scheduler.prepare_training_plan(_context(), config)

    assert plan.launch
    assert plan.execution_strategy is DrafterExecutionStrategy.SYNC
    assert plan.reason == "sync_fallback_training_ready"
    assert plan.max_batches == config.train_batches_per_trigger
    assert plan.training_group_id == "sync-fallback"

    event = scheduler.on_before_actor_update(
        BeforeActorUpdateContext(
            schedule_context=_context(),
            config=config,
        )
    )

    assert event.metrics is not None
    assert event.metrics["bubble/sync_fallback_requested"] == 1
    assert event.metrics["bubble/sync_fallback_launched"] == 1
    assert event.metrics["bubble/sync_fallback_batches"] == (
        config.train_batches_per_trigger
    )


def test_idle_worker_starvation_guard_is_disabled_for_background_poll() -> None:
    scheduler = _scheduler_with_statuses(("0", "1"))
    scheduler.on_worker_event(
        RolloutWorkerEvent(
            RolloutWorkerEventType.GENERATION_STARTED,
            worker_id="0",
            replica_rank=0,
        )
    )
    config = replace(
        _auto_idle_config(1),
        idle_worker_group_size=None,
        idle_worker_fallback_to_sync=True,
        max_steps_without_training=2,
    )

    plan = scheduler.prepare_training_plan(
        _context(),
        config,
        allow_sync_fallback=False,
    )

    assert not plan.launch
    assert plan.reason == "missing_training_group_metadata"


def test_sync_fallback_training_outcome_reports_bubble_metrics() -> None:
    state = DrafterRuntimeState()
    plan = TrainingPlan(
        launch=True,
        reason="sync_fallback_training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        data_version=10,
        required_target_version=10,
        plan_id="fallback-plan",
        worker_snapshots={
            "0": {
                "buffer_version": 1,
                "data_version": 10,
                "worker_incarnation": "worker-0",
                "trainable_samples": 5,
            }
        },
    )
    state.submit(plan, started_at=time.time())
    state.mark_running()

    outcome = TrainingOutcome.from_execution(
        ExecutionOutcome(
            raw_results=[
                {
                    "trained": True,
                    "triggered": True,
                    "source_global_step": 10,
                    "execution_strategy": "sync",
                    "attempted_steps": 1,
                    "successful_steps": 1,
                    "optimizer_step": 1,
                    "buffer_size_before": 5,
                    "buffer_size_after": 4,
                    "elapsed_sec": 0.5,
                    "reason": "trained",
                    "publish_snapshot_cached": True,
                    "worker_id": "0",
                    "worker_incarnation": "worker-0",
                    "plan_id": "fallback-plan",
                    "data_version": 10,
                    "target_version": 10,
                    "is_publish_leader": True,
                }
            ],
            elapsed_sec=0.5,
        ),
        runtime_state=state,
        plan=plan,
    )

    assert outcome.trained
    assert outcome.metrics["bubble/sync_fallback_completed"] == 1
    assert outcome.metrics["bubble/sync_fallback_successful_steps"] == 1
    assert outcome.metrics["bubble/sync_fallback_elapsed_s"] == 0.5


def test_idle_worker_starvation_guard_config_from_nested_mapping() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "max_steps_without_training": 20,
            "scheduler": {
                "idle_worker": {
                    "fallback_to_sync": True,
                    "max_seconds_without_training": 60.5,
                    "require_runtime_idle_events": True,
                },
            },
        }
    )

    assert config.idle_worker_fallback_to_sync is True
    assert config.idle_worker_require_runtime_idle_events is True
    assert config.max_steps_without_training == 20
    assert config.idle_worker_max_seconds_without_training == 60.5


def test_scheduler_duplicate_tuning_subtrees_do_not_override_training_fields() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 3,
            "publish_interval_steps": 4,
            "publish_async": False,
            "step": 5,
            "collection_sample_rate": 0.5,
            "max_collect_samples_per_step_per_replica": 6,
            "max_collect_tokens_per_step_per_replica": 7,
            "min_trainable_batches": 8,
            "max_steps_without_training": 9,
            "require_full_batch": False,
            "sample_last_n_steps": 10,
            "scheduler": {
                "collection": {
                    "interval_steps": 20,
                    "sample_rate": 0.1,
                    "max_samples_per_step_per_replica": 60,
                    "max_tokens_per_step_per_replica": 70,
                },
                "trigger": {
                    "interval_steps": 30,
                    "min_trainable_batches": 80,
                    "max_steps_without_training": 90,
                },
                "budget": {
                    "max_batches": 50,
                    "require_full_batch": True,
                    "sample_last_n_steps": 100,
                },
                "publish": {
                    "interval_optimizer_steps": 40,
                    "async_update": True,
                },
            },
        }
    )

    assert config.collect_interval_steps == 2
    assert config.training_interval_steps == 3
    assert config.publish_interval_steps == 4
    assert config.publish_async is False
    assert config.train_batches_per_trigger == 5
    assert config.collection_sample_rate == 0.5
    assert config.max_collect_samples_per_replica == 6
    assert config.max_collect_tokens_per_replica == 7
    assert config.min_trainable_batches == 8
    assert config.max_steps_without_training == 9
    assert config.require_full_batch is False
    assert config.sample_last_n_steps == 10


def test_idle_worker_plan_caps_batches_by_window_data_and_config() -> None:
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [
                {
                    **_status("0").__dict__,
                    "available": True,
                    "rank": 0,
                },
                {
                    **_status("1").__dict__,
                    "available": True,
                    "rank": 1,
                },
            ],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
        )
    )
    deadline_ts = time.time() + 2.8
    for worker_id in ("0", "1"):
        scheduler.on_worker_event(
            RolloutWorkerEvent(
                RolloutWorkerEventType.WORKER_IDLE,
                worker_id=worker_id,
                replica_rank=int(worker_id),
                memory_released=True,
                must_be_ready_at=deadline_ts,
                event_ts=100.0,
            )
        )

    plan = scheduler.prepare_training_plan(_context(), _idle_config())

    assert plan.launch
    assert plan.max_batches == 2
    assert plan.target_worker_ids == ("0", "1")
    assert plan.training_group_id == "idle-group-0"
    assert plan.to_worker_payload()["execution_strategy"] == "rollout_idle_worker"


def test_idle_worker_async_submit_and_poll_completion() -> None:
    events = []
    state = DrafterRuntimeState()
    plan = replace(
        DrafterScheduler().plan_training(
            DrafterScheduleContext(
                global_step=10,
                training_mode="online",
                collected_samples_this_step=2,
                oldlogprob_collection_requested=False,
                data_status=TrainingDataStatus(
                    **{
                        **_status("0").__dict__,
                        "worker_snapshots": {
                            "0": {
                                "buffer_version": 1,
                                "data_version": 10,
                                "worker_incarnation": "worker-0",
                                "trainable_samples": 5,
                            }
                        },
                    }
                ),
            ),
            _idle_config(),
            resources=type(
                "Resources",
                (),
                {
                    "worker_ids": ("0",),
                    "training_group_id": "idle-group-0",
                    "minimum_idle_window_sec": 3.0,
                },
            )(),
        ),
        max_batches=1,
    )
    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append(("submit", payload)) or "ref",
            resolve=lambda value: (
                [
                    {
                        "participating": True,
                        "ready": True,
                        "worker_id": "0",
                        "worker_incarnation": "worker-0",
                        "data_version": 10,
                        "target_version": 10,
                    }
                ]
                if value == "preflight-ref"
                else [
                    {
                        "trained": True,
                        "triggered": True,
                        "source_global_step": 10,
                        "execution_strategy": "rollout_idle_worker",
                        "attempted_steps": 1,
                        "successful_steps": 1,
                        "optimizer_step": 1,
                        "buffer_size_before": 5,
                        "buffer_size_after": 4,
                        "elapsed_sec": 0.8,
                        "reason": "trained",
                        "publish_snapshot_cached": True,
                        "worker_id": "0",
                        "worker_incarnation": "worker-0",
                        "plan_id": plan.plan_id,
                        "data_version": 10,
                        "target_version": 10,
                        "is_publish_leader": True,
                    }
                ]
            ),
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: "preflight-ref",
            abort_preflight=lambda plan_id: [],
            poll=lambda submission: (
                events.append(("poll", submission)) or (True, submission)
            ),
        )
    )

    execution = scheduler.execute_training_plan(plan, runtime_state=state)

    assert execution.reason == "submitted_async"
    assert state.status is DrafterRuntimeStatus.RUNNING
    assert events[0] == ("submit", plan.to_worker_payload())

    outcome = scheduler.poll_pending_training(runtime_state=state)

    assert outcome is not None
    assert outcome.trained
    assert outcome.successful_steps == 1
    assert scheduler._idle_worker_batch_estimate_sec == pytest.approx(0.8)
    assert state.status is DrafterRuntimeStatus.IDLE


def _trainer_with_idle_config() -> SpecoRayPPOTrainer:
    assert SpecoRayPPOTrainer is not None
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = {
        "actor_rollout_ref": {
            "rollout": {
                "data_parallel_size": 2,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "training": {
                        "scheduler": {
                            "execution": {"strategy": "rollout_idle_worker"},
                            "idle_worker": {
                                "training_groups": [["worker-0", "worker-1"]],
                                "initial_batch_estimate_sec": 0.5,
                                "max_batches_per_window": 2,
                                "deadline_guard_sec": 0.1,
                            },
                        }
                    },
                },
            }
        }
    }
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._drafter_runtime_state = DrafterRuntimeState()
    return trainer


def _trainer_with_default_idle_budget() -> SpecoRayPPOTrainer:
    assert SpecoRayPPOTrainer is not None
    trainer = SpecoRayPPOTrainer.__new__(SpecoRayPPOTrainer)
    trainer.config = {
        "actor_rollout_ref": {
            "rollout": {
                "data_parallel_size": 1,
                "drafter": {
                    "enable": True,
                    "enable_drafter_training": True,
                    "training": {
                        "scheduler": {
                            "execution": {"strategy": "rollout_idle_worker"},
                            "idle_worker": {
                                "training_groups": [["worker-0"]],
                            },
                        }
                    },
                },
            }
        }
    }
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._drafter_runtime_state = DrafterRuntimeState()
    return trainer


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_generation_completion_does_not_create_synthetic_idle_window() -> None:
    trainer = _trainer_with_idle_config()
    output = _FakeGenerationOutput(
        [{"replica_rank": 0, "id": "a"}, {"replica_rank": 1, "id": "b"}]
    )

    start_metrics = trainer._speco_emit_rollout_generation_started()
    complete_metrics = trainer._speco_emit_rollout_generation_completed(output)

    assert start_metrics["bubble/idle_workers"] == 0
    assert complete_metrics == {}
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 0
    assert output.non_tensor_batch["drafter_sample"][0]["id"] == "a"


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_fallback_idle_events_from_generation_output() -> None:
    trainer = _trainer_with_idle_config()
    output = _FakeGenerationOutput(
        [{"replica_rank": 0, "id": "a"}, {"replica_rank": 1, "id": "b"}]
    )

    trainer._speco_emit_rollout_generation_started()
    metrics = trainer._speco_emit_rollout_idle_from_generation_output(
        output,
        reason="test_no_runtime_events",
    )

    assert metrics["bubble/fallback_idle_events"] == 2
    assert metrics["bubble/idle_workers"] == 2
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 2
    resources = trainer._drafter_scheduler.select_idle_training_resources(
        trainer._speco_drafter_schedule_config()
    )
    assert resources.available is False
    assert resources.reason == "incomplete_training_group"
    assert output.non_tensor_batch["drafter_sample"][0]["id"] == "a"


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_strict_idle_mode_rejects_synthetic_idle_window() -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"][
        "scheduler"
    ]["idle_worker"]["require_runtime_idle_events"] = True
    output = _FakeGenerationOutput([{"replica_rank": 0, "id": "a"}])

    metrics = trainer._speco_emit_rollout_idle_from_generation_output(
        output,
        reason="test_no_runtime_events",
    )

    assert metrics == {"bubble/fallback_idle_events_disabled": 1}
    assert trainer._drafter_scheduler.idle_worker_metrics()["bubble/idle_workers"] == 0


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_fallback_idle_uses_bootstrap_window_floor() -> None:
    trainer = _trainer_with_default_idle_budget()

    deadline_ts = trainer._speco_rollout_idle_fallback_deadline_ts()

    assert deadline_ts - time.time() >= 9.0


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_reclaims_active_idle_workers_before_next_generation() -> None:
    trainer = _trainer_with_idle_config()
    events = []
    trainer._drafter_scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
            reclaim=lambda worker_ids: events.append(worker_ids),
        )
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("worker-0", "worker-1"),
    )
    trainer._drafter_runtime_state.submit(plan, started_at=time.time())
    trainer._drafter_runtime_state.mark_running()
    drain_calls = []
    trainer._speco_wait_pending_drafter_training = lambda: (
        drain_calls.append(True) or (plan, None)
    )

    metrics = trainer._speco_reclaim_rollout_idle_workers_before_generation()

    assert metrics["bubble/reclaim_requested"] == 1
    assert metrics["bubble/reclaim_drained"] == 0
    assert metrics["timing_s/drafter_reclaim_wait"] >= 0.0
    assert metrics["timing_s/drafter_critical_path_before_generation"] >= 0.0
    assert events == [("worker-0", "worker-1")]
    assert drain_calls == [True]
    assert len(trainer._drafter_scheduler._idle_worker_reclaim_samples_sec) == 1


@pytest.mark.skipif(SpecoRayPPOTrainer is None, reason="ray/verl is not installed")
def test_trainer_can_skip_reclaim_drain_when_configured() -> None:
    trainer = _trainer_with_idle_config()
    trainer.config["actor_rollout_ref"]["rollout"]["drafter"]["training"][
        "scheduler"
    ]["idle_worker"]["drain_before_next_rollout"] = False
    events = []
    trainer._drafter_scheduler.bind_worker_executor(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: None,
            resolve=lambda value: value,
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
            activate=lambda: [],
            preflight=lambda payload: [],
            abort_preflight=lambda plan_id: [],
            reclaim=lambda worker_ids: events.append(worker_ids),
        )
    )
    plan = TrainingPlan(
        launch=True,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
        source_global_step=10,
        max_batches=1,
        publish_after_success=True,
        target_worker_ids=("worker-0", "worker-1"),
    )
    trainer._drafter_runtime_state.submit(plan, started_at=time.time())
    trainer._drafter_runtime_state.mark_running()
    drain_calls = []
    trainer._speco_wait_pending_drafter_training = lambda: (
        drain_calls.append(True) or (plan, None)
    )

    metrics = trainer._speco_reclaim_rollout_idle_workers_before_generation()

    assert metrics == {"bubble/reclaim_requested": 1}
    assert events == [("worker-0", "worker-1")]
    assert drain_calls == []
