# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

import pytest

from verl_speco.trainer.scheduler import (
    CallbackDrafterWorkerExecutor,
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    DrafterRuntimeStatus,
    DrafterScheduler,
    TrainingPlan,
)


def _plan(*, launch: bool = True) -> TrainingPlan:
    return TrainingPlan(
        launch=launch,
        reason="training_ready",
        interval_matched=True,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=4,
        max_batches=2,
        publish_after_success=True,
    )


def test_sync_execution_submits_payload_and_blocks_for_results() -> None:
    events = []
    state = DrafterRuntimeState()

    scheduler = DrafterScheduler(
        CallbackDrafterWorkerExecutor(
            submit=lambda payload: events.append(("submit", payload)) or "ref",
            resolve=lambda ref: events.append(("resolve", ref))
            or [{"trained": True}],
            inspect_data=lambda sample_last_n_steps, require_full_batch: [],
            prepare=lambda plan: {},
        )
    )
    outcome = scheduler.execute_training_plan(
        _plan(),
        runtime_state=state,
    )

    assert events[0][0] == "submit"
    assert events[0][1]["max_batches"] == 2
    assert events[1] == ("resolve", "ref")
    assert outcome.raw_results == [{"trained": True}]
    assert state.status is DrafterRuntimeStatus.RUNNING


def test_sync_execution_marks_runtime_failed() -> None:
    state = DrafterRuntimeState()
    with pytest.raises(RuntimeError, match="rpc failed"):
        DrafterScheduler(
            CallbackDrafterWorkerExecutor(
                submit=lambda payload: "ref",
                resolve=lambda ref: (_ for _ in ()).throw(RuntimeError("rpc failed")),
                inspect_data=lambda sample_last_n_steps, require_full_batch: [],
                prepare=lambda plan: {},
            )
        ).execute_training_plan(
            _plan(),
            runtime_state=state,
        )
    assert state.status is DrafterRuntimeStatus.FAILED


def test_sync_execution_rejects_inactive_plan() -> None:
    with pytest.raises(ValueError):
        DrafterScheduler(
            CallbackDrafterWorkerExecutor(
                submit=lambda payload: None,
                resolve=lambda ref: None,
                inspect_data=lambda sample_last_n_steps, require_full_batch: [],
                prepare=lambda plan: {},
            )
        ).execute_training_plan(
            _plan(launch=False),
            runtime_state=DrafterRuntimeState(),
        )


def test_execution_requires_bound_worker_executor() -> None:
    with pytest.raises(RuntimeError, match="has not been bound"):
        DrafterScheduler().execute_training_plan(
            _plan(), runtime_state=DrafterRuntimeState()
        )
