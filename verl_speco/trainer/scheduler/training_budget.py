# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training budget policies for drafter scheduling."""

from __future__ import annotations

from typing import Protocol

from verl_speco.trainer.scheduler.schedule_types import (
    DrafterScheduleConfig,
    DrafterScheduleContext,
    TrainingBudget,
)


class TrainingBudgetPolicy(Protocol):
    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget: ...


class SyncTrainingBudgetPolicy:
    """Bound synchronous work by configuration and actually trainable batches."""

    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget:
        status = context.data_status
        trainable_batches = status.trainable_batches if status is not None else 0
        max_batches = min(max(config.train_batches_per_trigger, 0), trainable_batches)
        return TrainingBudget(
            max_batches=max_batches,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            reason="sync_budget_ready" if max_batches > 0 else "no_training_budget",
        )
