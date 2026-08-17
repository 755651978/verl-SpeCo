# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from unittest.mock import Mock

from verl_speco.trainer.scheduler import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    PublishOutcome,
    PublishPlan,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome


def _training_plan(*, launch: bool = True) -> TrainingPlan:
    return TrainingPlan(
        launch=launch,
        reason="training_ready" if launch else "interval_not_reached",
        interval_matched=launch,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=4,
        max_batches=1 if launch else 0,
        publish_after_success=True,
    )


def test_before_actor_update_plans_then_prepares() -> None:
    scheduler = DrafterScheduler()
    plan = _training_plan()
    scheduler.prepare_training_plan = Mock(return_value=plan)
    scheduler.prepare_training_execution = Mock(
        return_value={"drafter/target_lm_head_synced": 1}
    )
    context = BeforeActorUpdateContext(
        schedule_context=DrafterScheduleContext(
            global_step=4,
            training_mode="online",
            collected_samples_this_step=1,
            oldlogprob_collection_requested=False,
        ),
        config=DrafterScheduleConfig(),
    )

    outcome = scheduler.on_before_actor_update(context)

    assert outcome.training_plan is plan
    assert outcome.metrics["drafter/target_lm_head_synced"] == 1
    scheduler.prepare_training_plan.assert_called_once()
    scheduler.prepare_training_execution.assert_called_once_with(plan)


def test_after_actor_update_executes_prepared_plan() -> None:
    scheduler = DrafterScheduler()
    execution = ExecutionOutcome(
        raw_results=[
            {
                "trained": True,
                "successful_steps": 1,
                "attempted_steps": 1,
            }
        ],
        elapsed_sec=0.5,
    )
    scheduler.execute_training_plan = Mock(return_value=execution)
    plan = _training_plan()
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert outcome.training_execution.trained
    assert outcome.training_execution.successful_steps == 1
    scheduler.execute_training_plan.assert_called_once_with(
        plan, runtime_state=runtime_state
    )


def test_after_actor_update_skips_inactive_plan() -> None:
    scheduler = DrafterScheduler()
    scheduler.execute_training_plan = Mock()

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(_training_plan(launch=False), DrafterRuntimeState())
    )

    assert outcome.training_execution is None
    scheduler.execute_training_plan.assert_not_called()


def test_after_actor_update_normalizes_preflight_failure_and_resets_runtime() -> None:
    scheduler = DrafterScheduler()
    plan = _training_plan()
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[{"ready": False, "reason": "buffer_version_changed"}],
            elapsed_sec=0.1,
            launched=False,
            reason="worker_preflight_failed",
        )
    )

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert not outcome.training_execution.trained
    assert outcome.training_execution.reason == "worker_preflight_failed"
    assert runtime_state.status.name == "IDLE"


def test_after_weight_update_plans_then_publishes() -> None:
    scheduler = DrafterScheduler()
    publish_plan = PublishPlan(
        publish=True,
        reason="publish_interval_reached",
        interval_matched=True,
        source_global_step=4,
    )
    publish_outcome = PublishOutcome(attempted=True, published=True)
    scheduler.plan_publish = Mock(return_value=publish_plan)
    scheduler.execute_publish_plan = Mock(return_value=publish_outcome)
    training_plan = _training_plan()

    outcome = scheduler.on_after_weight_update(
        AfterWeightUpdateContext(
            global_step=4,
            drafter_trained=True,
            config=DrafterScheduleConfig(),
            training_plan=training_plan,
        )
    )

    assert outcome.publish_plan is publish_plan
    assert outcome.publish_outcome is publish_outcome
    assert outcome.metrics["drafter/published"] == 1
    scheduler.execute_publish_plan.assert_called_once_with(publish_plan)
