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
from __future__ import annotations

import pytest

from verl_speco.trainer.drafter_scheduler import (
    DrafterScheduler,
    step_matches_interval,
)
from verl_speco.trainer.schedule_types import (
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
)


def _context(
    *,
    step=5,
    mode="online",
    samples=1,
    oldlogprob_requested=False,
) -> DrafterScheduleContext:
    return DrafterScheduleContext(
        global_step=step,
        training_mode=mode,
        collected_samples_this_step=samples,
        oldlogprob_collection_requested=oldlogprob_requested,
    )


def test_step_interval_matches_released_semantics() -> None:
    assert step_matches_interval(6, 3)
    assert not step_matches_interval(0, 3)
    assert not step_matches_interval(None, 3)
    assert not step_matches_interval(6, 0)
    assert not step_matches_interval("bad-step", 3)
    assert not step_matches_interval(6, "bad-interval")


def test_legacy_config_maps_released_sync_values() -> None:
    config = DrafterScheduleConfig.from_mapping(
        {
            "collect_interval_steps": 2,
            "training_interval_steps": 5,
            "publish_interval_steps": 10,
            "use_data_buffer": True,
            "step": 7,
        }
    )
    assert config == DrafterScheduleConfig(
        collect_interval_steps=2,
        training_interval_steps=5,
        publish_interval_steps=10,
        use_data_buffer=True,
        train_batches_per_trigger=7,
    )


def test_sync_plan_launches_for_current_step_samples() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(training_interval_steps=5)

    plan = scheduler.plan_training(_context(), config)

    assert plan.launch
    assert plan.reason == "current_step_samples"
    assert plan.interval_matched
    assert plan.execution_strategy is DrafterExecutionStrategy.SYNC
    assert plan.source_global_step == 5
    assert plan.publish_after_success
    assert plan.to_worker_payload()["execution_strategy"] == "sync"


@pytest.mark.parametrize(
    ("context", "config", "reason"),
    [
        (_context(mode="collect_only"), DrafterScheduleConfig(), "collect_only"),
        (
            _context(step=4),
            DrafterScheduleConfig(training_interval_steps=5),
            "interval_not_reached",
        ),
        (
            _context(samples=0, oldlogprob_requested=True),
            DrafterScheduleConfig(training_interval_steps=5, use_data_buffer=True),
            "no_current_step_oldlogprob_samples",
        ),
        (
            _context(samples=0),
            DrafterScheduleConfig(training_interval_steps=5),
            "no_current_step_samples",
        ),
    ],
)
def test_sync_plan_preserves_skip_conditions(context, config, reason) -> None:
    plan = DrafterScheduler().plan_training(context, config)
    assert not plan.launch
    assert plan.reason == reason


def test_sync_plan_preserves_data_buffer_fallback() -> None:
    plan = DrafterScheduler().plan_training(
        _context(samples=0),
        DrafterScheduleConfig(
            training_interval_steps=5,
            use_data_buffer=True,
            train_batches_per_trigger=9,
        ),
    )
    assert plan.launch
    assert plan.reason == "data_buffer_enabled"
    assert plan.max_batches == 9
    assert plan.publish_after_success


def test_sync_plan_carries_publish_decision_to_worker() -> None:
    plan = DrafterScheduler().plan_training(
        _context(step=6),
        DrafterScheduleConfig(
            training_interval_steps=3,
            publish_interval_steps=4,
        ),
    )
    assert plan.launch
    assert not plan.publish_after_success


def test_publish_plan_preserves_released_interval_behavior() -> None:
    scheduler = DrafterScheduler()
    config = DrafterScheduleConfig(publish_interval_steps=4)

    assert not scheduler.plan_publish(
        global_step=6, drafter_trained=True, config=config
    ).publish
    assert scheduler.plan_publish(
        global_step=8, drafter_trained=True, config=config
    ).publish
    assert not scheduler.plan_publish(
        global_step=8, drafter_trained=False, config=config
    ).publish


def test_invalid_publish_interval_still_raises() -> None:
    with pytest.raises(ValueError):
        DrafterScheduler().plan_publish(
            global_step=5,
            drafter_trained=True,
            config=DrafterScheduleConfig(publish_interval_steps="bad"),
        )
