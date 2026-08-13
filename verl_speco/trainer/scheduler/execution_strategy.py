# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Execution strategies for drafter training plans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.schedule_types import TrainingPlan
from verl_speco.trainer.scheduler.worker_executor import DrafterWorkerExecutor


@dataclass(frozen=True)
class ExecutionOutcome:
    raw_results: list[Any]
    elapsed_sec: float


class DrafterTrainingExecutionStrategy(Protocol):
    def execute(
        self,
        plan: TrainingPlan,
        *,
        executor: DrafterWorkerExecutor,
        runtime_state: DrafterRuntimeState,
    ) -> ExecutionOutcome: ...


class SyncExecutionStrategy:
    """Submit a training plan and synchronously wait for all worker results."""

    def execute(
        self,
        plan: TrainingPlan,
        *,
        executor: DrafterWorkerExecutor,
        runtime_state: DrafterRuntimeState,
    ) -> ExecutionOutcome:
        if not plan.launch:
            raise ValueError("Cannot execute an inactive drafter training plan")
        if runtime_state.status in {
            DrafterRuntimeStatus.COMPLETED,
            DrafterRuntimeStatus.FAILED,
        }:
            runtime_state.reset()

        started_at = time.perf_counter()
        runtime_state.submit(plan, started_at=started_at)
        runtime_state.mark_running()
        try:
            submission = executor.submit_training(plan)
            results = executor.resolve_training(submission)
        except Exception as error:
            runtime_state.mark_failed(error)
            raise
        elapsed_sec = time.perf_counter() - started_at
        return ExecutionOutcome(raw_results=results, elapsed_sec=elapsed_sec)
