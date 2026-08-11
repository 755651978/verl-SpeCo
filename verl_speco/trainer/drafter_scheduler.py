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
"""Legacy-equivalent synchronous drafter scheduling.

This module is the single decision source for synchronous collection, training,
batch limits, and publication. Trainers and runtimes request plans; workers only
execute the serialized plan. The blocking training behavior and call ordering
remain unchanged.
"""

from __future__ import annotations

from typing import Any

from verl_speco.trainer.schedule_types import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    PublishPlan,
    TrainingPlan,
)


def step_matches_interval(
    global_step: Any,
    interval_steps: Any,
    *,
    default_interval: int = 1,
) -> bool:
    """Match the released ``speco_step_matches_interval`` semantics exactly."""

    try:
        interval = int(default_interval if interval_steps is None else interval_steps)
    except (TypeError, ValueError):
        return False
    if interval <= 0 or global_step is None:
        return False
    try:
        step = int(global_step)
    except (TypeError, ValueError):
        return False
    return step > 0 and step % interval == 0


class DrafterScheduler:
    """Make synchronous collect, train, and publish decisions.

    The class is intentionally stateless in PR 1. Every decision receives a
    fresh legacy-compatible config snapshot so runtime config mutation retains
    the same behavior as the released trainer implementation.
    """

    @staticmethod
    def should_collect(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.collect_interval_steps)

    @staticmethod
    def training_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.training_interval_steps)

    def plan_training(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        common = {
            "interval_matched": interval_matched,
            "execution_strategy": DrafterExecutionStrategy.SYNC,
            "source_global_step": context.global_step,
            "max_batches": config.train_batches_per_trigger,
        }
        if context.training_mode == "collect_only":
            return TrainingPlan(
                launch=False,
                reason="collect_only",
                publish_after_success=False,
                **common,
            )
        if not interval_matched:
            return TrainingPlan(
                launch=False,
                reason="interval_not_reached",
                publish_after_success=False,
                **common,
            )
        if context.collected_samples_this_step > 0:
            return TrainingPlan(
                launch=True,
                reason="current_step_samples",
                publish_after_success=self._publish_interval_matched(
                    context.global_step, config
                ),
                **common,
            )
        if context.oldlogprob_collection_requested:
            return TrainingPlan(
                launch=False,
                reason="no_current_step_oldlogprob_samples",
                publish_after_success=False,
                **common,
            )
        if config.use_data_buffer:
            return TrainingPlan(
                launch=True,
                reason="data_buffer_enabled",
                publish_after_success=self._publish_interval_matched(
                    context.global_step, config
                ),
                **common,
            )
        return TrainingPlan(
            launch=False,
            reason="no_current_step_samples",
            publish_after_success=False,
            **common,
        )

    @staticmethod
    def _publish_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        interval = int(config.publish_interval_steps or 0)
        return interval <= 0 or global_step % interval == 0

    @staticmethod
    def plan_publish(
        *,
        global_step: object,
        drafter_trained: bool,
        config: DrafterScheduleConfig,
    ) -> PublishPlan:
        if not drafter_trained:
            return PublishPlan(
                publish=False,
                reason="drafter_not_trained",
                interval_matched=False,
                source_global_step=global_step,
            )
        # Preserve the released path exactly: invalid publish configuration is
        # an error instead of being silently converted into a skipped publish.
        interval_matched = DrafterScheduler._publish_interval_matched(
            global_step, config
        )
        return PublishPlan(
            publish=interval_matched,
            reason=(
                "publish_interval_reached"
                if interval_matched
                else "publish_interval_not_reached"
            ),
            interval_matched=interval_matched,
            source_global_step=global_step,
        )
