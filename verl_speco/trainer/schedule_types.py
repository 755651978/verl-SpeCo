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
"""Pure scheduling contracts for online drafter training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DrafterExecutionStrategy(str, Enum):
    """Supported drafter-training execution strategies.

    PR 1 intentionally executes only ``SYNC``. ``ROLLOUT_IDLE_WORKER`` is
    reserved in the contract so the later bubble-time implementation can reuse
    the same plan type without changing the released synchronous path.
    """

    SYNC = "sync"
    ROLLOUT_IDLE_WORKER = "rollout_idle_worker"


@dataclass(frozen=True)
class DrafterScheduleConfig:
    """Legacy-compatible scheduling values read from ``training`` config."""

    collect_interval_steps: object = 1
    training_interval_steps: object = 1
    publish_interval_steps: object = 0
    use_data_buffer: bool = False
    train_batches_per_trigger: int = 100

    @classmethod
    def from_mapping(cls, config) -> "DrafterScheduleConfig":
        config = config or {}
        get = config.get if hasattr(config, "get") else lambda key, default: default
        train_batches = int(get("step", 100))
        return cls(
            collect_interval_steps=get("collect_interval_steps", 1),
            training_interval_steps=get("training_interval_steps", 1),
            publish_interval_steps=get("publish_interval_steps", 0),
            use_data_buffer=bool(get("use_data_buffer", False)),
            train_batches_per_trigger=train_batches,
        )


@dataclass(frozen=True)
class DrafterScheduleContext:
    """Read-only facts used to make the released synchronous decision."""

    global_step: object
    training_mode: str
    collected_samples_this_step: int
    oldlogprob_collection_requested: bool


@dataclass(frozen=True)
class TrainingPlan:
    """The synchronous training decision produced by ``DrafterScheduler``."""

    launch: bool
    reason: str
    interval_matched: bool
    execution_strategy: DrafterExecutionStrategy
    source_global_step: object
    max_batches: int
    publish_after_success: bool

    def to_worker_payload(self) -> dict[str, object]:
        """Serialize the scheduler decision for the drafter worker RPC."""

        return {
            "launch": self.launch,
            "execution_strategy": self.execution_strategy.value,
            "source_global_step": self.source_global_step,
            "max_batches": self.max_batches,
            "publish_after_success": self.publish_after_success,
        }


@dataclass(frozen=True)
class PublishPlan:
    """Whether the already-released synchronous path should publish weights."""

    publish: bool
    reason: str
    interval_matched: bool
    source_global_step: object
