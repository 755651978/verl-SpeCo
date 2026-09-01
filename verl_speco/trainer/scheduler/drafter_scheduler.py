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

import logging
import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from verl_speco.trainer.scheduler.schedule_types import (
    CollectionPlan,
    CollectionPayload,
    AvailableTrainingResources,
    DrafterCollectionContext,
    DrafterCollectionSource,
    DrafterExecutionStrategy,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    RolloutWorkerEvent,
    RolloutWorkerEventType,
    PublishPlan,
    TrainingBudget,
    TrainingPlan,
    _as_int,
)
from verl_speco.trainer.scheduler.execution_strategy import (
    RolloutIdleWorkerExecutionStrategy,
    SyncExecutionStrategy,
)
from verl_speco.trainer.scheduler.training_budget import SyncTrainingBudgetPolicy
from verl_speco.trainer.scheduler.training_trigger import IntervalAndBufferTrigger
from verl_speco.trainer.scheduler.worker_executor import DrafterWorkerExecutor
from verl_speco.trainer.scheduler.publish_executor import DrafterPublishExecutor
from verl_speco.trainer.scheduler.publish_strategy import PublishExecutionStrategy
from verl_speco.trainer.scheduler.data_status_policy import (
    ConservativeTrainingDataStatusPolicy,
)
from verl_speco.trainer.scheduler.lifecycle import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    SchedulerEventOutcome,
)
from verl_speco.trainer.scheduler.collection_executor import (
    DrafterCollectionExecutor,
)
from verl_speco.trainer.scheduler.collection_strategy import SyncCollectionStrategy
from verl_speco.trainer.scheduler.collection_adapter import (
    DrafterCollectionAdapter,
    OldLogProbCollectionAdapter,
    SGLangCollectionAdapter,
)
from verl_speco.trainer.scheduler.training_outcome import TrainingOutcome

logger = logging.getLogger(__name__)

_BOOTSTRAP_IDLE_BATCH_ESTIMATE_SEC = 0.25
_BOOTSTRAP_IDLE_DEADLINE_GUARD_SEC = 0.05
_BOOTSTRAP_IDLE_STARTUP_RESERVE_SEC = 2.0


def _conservative_percentile(
    values: Iterable[float], quantile: float = 0.9
) -> float:
    samples = sorted(max(float(value), 0.0) for value in values)
    if not samples:
        return 0.0
    index = max(0, min(int(math.ceil(quantile * len(samples))) - 1, len(samples) - 1))
    return samples[index]


@dataclass
class _IdleWorkerState:
    worker_id: str
    replica_rank: int
    status: str = "ready"
    memory_released: bool = False
    must_be_ready_at: float | None = None
    event_ts: float = 0.0


def _rollout_worker_event_type(value: object) -> RolloutWorkerEventType:
    if isinstance(value, RolloutWorkerEventType):
        return value
    text = str(value)
    try:
        return RolloutWorkerEventType(text)
    except ValueError:
        return RolloutWorkerEventType[text.upper()]


def _natural_worker_sort_key(worker_id: object) -> tuple[str, int, str]:
    """Sort worker ids by trailing numeric rank when possible."""

    text = str(worker_id)
    prefix, separator, suffix = text.rpartition("-")
    if separator and suffix.isdigit():
        return (prefix, int(suffix), text)
    if text.isdigit():
        return ("", int(text), text)
    return (text, -1, text)


def _normalize_worker_id_group(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    try:
        worker_ids = [str(worker_id) for worker_id in value]  # type: ignore[union-attr]
    except TypeError:
        return (str(value),)
    return tuple(dict.fromkeys(sorted(worker_ids, key=_natural_worker_sort_key)))


def _flatten_metadata_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        records: list[dict[str, Any]] = []
        for item in value:
            records.extend(_flatten_metadata_records(item))
        return records
    return []


def _idle_state_summary(
    states: dict[str, _IdleWorkerState],
    *,
    now: float | None = None,
) -> list[dict[str, object]]:
    now = time.time() if now is None else now
    summary: list[dict[str, object]] = []
    for worker_id in sorted(states, key=_natural_worker_sort_key):
        state = states[worker_id]
        window = (
            None
            if state.must_be_ready_at is None
            else max(float(state.must_be_ready_at) - now, 0.0)
        )
        summary.append(
            {
                "worker_id": worker_id,
                "replica_rank": state.replica_rank,
                "status": state.status,
                "memory_released": state.memory_released,
                "window_s": None if window is None else round(window, 3),
                "event_age_s": round(max(now - float(state.event_ts), 0.0), 3),
            }
        )
    return summary


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

    Every decision receives a
    fresh legacy-compatible config snapshot so runtime config mutation retains
    the same behavior as the released trainer implementation.
    """

    def __init__(
        self,
        worker_executor: DrafterWorkerExecutor | None = None,
        publish_executor: DrafterPublishExecutor | None = None,
        collection_executor: DrafterCollectionExecutor | None = None,
    ) -> None:
        self.trigger_policy = IntervalAndBufferTrigger()
        self.sync_budget_policy = SyncTrainingBudgetPolicy()
        self.sync_execution_strategy = SyncExecutionStrategy()
        self._worker_executor = worker_executor
        self.data_status_policy = ConservativeTrainingDataStatusPolicy()
        self.publish_execution_strategy = PublishExecutionStrategy()
        self._publish_executor = publish_executor
        self.collection_strategy = SyncCollectionStrategy()
        self._collection_executor = collection_executor
        self._collection_adapters: dict[
            DrafterCollectionSource, DrafterCollectionAdapter
        ] = {
            DrafterCollectionSource.SGLANG: SGLangCollectionAdapter(),
            DrafterCollectionSource.OLD_LOGPROB: OldLogProbCollectionAdapter(),
        }
        self.rollout_idle_execution_strategy = RolloutIdleWorkerExecutionStrategy()
        self._idle_workers: dict[str, _IdleWorkerState] = {}
        self._metadata_idle_training_groups: tuple[tuple[str, ...], ...] = ()
        self._replica_idle_worker_groups: dict[int, tuple[str, ...]] = {}
        self._last_successful_training_step: int | None = None
        self._last_successful_training_ts: float | None = None
        self._training_progress_start_ts: float = time.time()
        self._idle_worker_batch_estimate_sec: float | None = None
        self._idle_worker_batch_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_worker_reclaim_samples_sec: deque[float] = deque(maxlen=32)
        self._idle_worker_startup_samples_sec: deque[float] = deque(maxlen=32)
        self._replica_idle_started_at: dict[int, float] = {}
        self._replica_idle_window_samples_sec: deque[float] = deque(maxlen=32)

    def _effective_idle_batch_estimate_sec(
        self,
        config: DrafterScheduleConfig,
    ) -> float:
        if config.idle_worker_initial_batch_estimate_sec is not None:
            return max(float(config.idle_worker_initial_batch_estimate_sec), 1.0e-9)
        if self._idle_worker_batch_estimate_sec is not None:
            conservative_batch_sec = _conservative_percentile(
                self._idle_worker_batch_samples_sec
            )
            return max(
                float(self._idle_worker_batch_estimate_sec),
                conservative_batch_sec,
                1.0e-9,
            )
        return _BOOTSTRAP_IDLE_BATCH_ESTIMATE_SEC

    def _idle_batch_estimate_is_bootstrap(
        self,
        config: DrafterScheduleConfig,
    ) -> bool:
        return (
            config.idle_worker_initial_batch_estimate_sec is None
            and self._idle_worker_batch_estimate_sec is None
        )

    def _effective_idle_deadline_guard_sec(
        self,
        config: DrafterScheduleConfig,
    ) -> float:
        if config.idle_worker_deadline_guard_sec is not None:
            configured_guard = max(float(config.idle_worker_deadline_guard_sec), 0.0)
        else:
            estimate = self._effective_idle_batch_estimate_sec(config)
            configured_guard = max(
                _BOOTSTRAP_IDLE_DEADLINE_GUARD_SEC, estimate * 0.1
            )
        reclaim_guard = _conservative_percentile(
            self._idle_worker_reclaim_samples_sec
        )
        return max(configured_guard, reclaim_guard)

    def _effective_idle_startup_reserve_sec(
        self, config: DrafterScheduleConfig
    ) -> float:
        """Reserve activation/cleanup work before starting a Bubble batch."""

        historical = _conservative_percentile(self._idle_worker_startup_samples_sec)
        if historical > 0.0:
            return historical
        if self._idle_batch_estimate_is_bootstrap(config):
            return _BOOTSTRAP_IDLE_STARTUP_RESERVE_SEC
        return 0.0

    def record_reclaim_elapsed(self, elapsed_sec: float) -> None:
        elapsed_sec = max(float(elapsed_sec), 0.0)
        if elapsed_sec <= 0.0:
            return
        self._idle_worker_reclaim_samples_sec.append(elapsed_sec)
        logger.warning(
            "[BubbleTime] updated reclaim guard: observed_s=%.3f guard_s=%.3f samples=%s",
            elapsed_sec,
            _conservative_percentile(self._idle_worker_reclaim_samples_sec),
            len(self._idle_worker_reclaim_samples_sec),
        )

    def record_idle_training_outcome(self, outcome: TrainingOutcome) -> None:
        """Learn conservative admission overhead, including empty plans."""

        if outcome.successful_steps > 0:
            observed_sec = float(
                outcome.metrics.get("timing_s/drafter_worker_cleanup", 0.0) or 0.0
            )
        else:
            # An empty plan records exactly the cost that prevented batch 1
            # from starting (activation, reclaim, and cleanup included).
            observed_sec = max(float(outcome.elapsed_sec), 0.0)
        if observed_sec <= 0.0:
            return
        self._idle_worker_startup_samples_sec.append(observed_sec)
        logger.warning(
            "[BubbleTime] updated startup reserve: reason=%s observed_s=%.3f "
            "reserve_s=%.3f samples=%s",
            outcome.reason,
            observed_sec,
            _conservative_percentile(self._idle_worker_startup_samples_sec),
            len(self._idle_worker_startup_samples_sec),
        )

    def _effective_historical_idle_window_sec(self) -> float | None:
        if not self._replica_idle_window_samples_sec:
            return None
        # Use the lower tail.  A deadline based on a long-tail average would
        # start work that cannot be reclaimed before shorter rollout cycles.
        return _conservative_percentile(
            self._replica_idle_window_samples_sec,
            quantile=0.1,
        )

    def _effective_idle_min_window_sec(
        self,
        config: DrafterScheduleConfig,
    ) -> float:
        if config.idle_worker_min_idle_window_sec is not None:
            return max(float(config.idle_worker_min_idle_window_sec), 0.0)
        return self._effective_idle_deadline_guard_sec(config)

    def _effective_idle_max_batches_per_window(
        self,
        config: DrafterScheduleConfig,
    ) -> int:
        if config.idle_worker_max_batches_per_window is not None:
            return max(int(config.idle_worker_max_batches_per_window), 1)
        if self._idle_batch_estimate_is_bootstrap(config):
            return 1
        return max(int(config.train_batches_per_trigger), 1)

    def bind_worker_executor(self, worker_executor: DrafterWorkerExecutor) -> None:
        """Bind the worker execution port used by all execution strategies."""

        self._worker_executor = worker_executor

    def bind_publish_executor(self, publish_executor: DrafterPublishExecutor) -> None:
        self._publish_executor = publish_executor

    def bind_collection_executor(
        self, collection_executor: DrafterCollectionExecutor
    ) -> None:
        self._collection_executor = collection_executor

    def register_collection_adapter(self, adapter: DrafterCollectionAdapter) -> None:
        """Register or replace the payload adapter for a collection source."""
        self._collection_adapters[adapter.source] = adapter

    def prepare_collection_payload(
        self,
        *,
        source: DrafterCollectionSource,
        samples: list[dict],
        owner_count: int,
        dispatch_bucket_count: int | None,
        raw_samples: int,
        collection_id: str = "",
        owners=None,
    ) -> CollectionPayload:
        """Build the common payload without exposing bucketing to the Trainer."""
        adapter = self._collection_adapters.get(source)
        if adapter is None:
            raise ValueError(f"No collection adapter registered for {source.value}")
        return adapter.prepare_payload(
            samples,
            owner_count=owner_count,
            dispatch_bucket_count=dispatch_bucket_count,
            raw_samples=raw_samples,
            collection_id=collection_id,
            owners=owners,
        )

    def inspect_training_data(
        self,
        *,
        global_step: object,
        config: DrafterScheduleConfig,
        worker_ids: tuple[str, ...] | None = None,
    ):
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        statuses = self._worker_executor.get_training_data_status(
            sample_last_n_steps=config.sample_last_n_steps,
            require_full_batch=config.require_full_batch,
            worker_ids=worker_ids,
        )
        return self.data_status_policy.aggregate(statuses, global_step=global_step)

    def on_worker_event(
        self,
        event: RolloutWorkerEvent | dict[str, object],
    ) -> dict[str, float | int]:
        """Record rollout replica state for Bubble Time idle-worker planning."""

        if isinstance(event, dict):
            event = RolloutWorkerEvent(
                event_type=_rollout_worker_event_type(event.get("event_type", "")),
                worker_id=str(event.get("worker_id", "")),
                replica_rank=_as_int(event.get("replica_rank", 0)),
                memory_released=bool(event.get("memory_released", False)),
                release_source=str(event.get("release_source", "") or ""),
                must_be_ready_at=(
                    None
                    if event.get("must_be_ready_at") is None
                    else float(event.get("must_be_ready_at", 0.0))
                ),
                event_ts=(
                    None
                    if event.get("event_ts") is None
                    else float(event.get("event_ts", 0.0))
                ),
            )
        elif not isinstance(event.event_type, RolloutWorkerEventType):
            event = RolloutWorkerEvent(
                event_type=_rollout_worker_event_type(event.event_type),
                worker_id=event.worker_id,
                replica_rank=event.replica_rank,
                memory_released=event.memory_released,
                release_source=event.release_source,
                must_be_ready_at=event.must_be_ready_at,
                event_ts=event.event_ts,
            )
        event_ts = event.event_ts if event.event_ts is not None else time.time()
        if event.event_type is RolloutWorkerEventType.WORKER_IDLE:
            if event.memory_released:
                self._replica_idle_started_at[event.replica_rank] = event_ts
            else:
                self._replica_idle_started_at.pop(event.replica_rank, None)
        elif event.event_type is RolloutWorkerEventType.GENERATION_STARTED:
            idle_started_at = self._replica_idle_started_at.pop(
                event.replica_rank, None
            )
            if idle_started_at is not None and event_ts > idle_started_at:
                observed_window = event_ts - idle_started_at
                self._replica_idle_window_samples_sec.append(observed_window)
                logger.warning(
                    "[BubbleTime] observed replica idle window: replica_rank=%s "
                    "window_s=%.3f conservative_window_s=%.3f samples=%s",
                    event.replica_rank,
                    observed_window,
                    self._effective_historical_idle_window_sec(),
                    len(self._replica_idle_window_samples_sec),
                )
        worker_ids = self._replica_idle_worker_groups.get(event.replica_rank)
        if not worker_ids:
            worker_ids = (event.worker_id,)
        for worker_id in worker_ids:
            self._record_worker_event_state(event, worker_id, event_ts)
        logger.warning(
            "[BubbleTime] worker_event type=%s worker_id=%s replica_rank=%s "
            "expanded_worker_ids=%s memory_released=%s release_source=%s must_be_ready_at=%s "
            "event_ts=%s idle_state=%s",
            event.event_type.value,
            event.worker_id,
            event.replica_rank,
            worker_ids,
            event.memory_released,
            event.release_source,
            event.must_be_ready_at,
            event_ts,
            _idle_state_summary(self._idle_workers, now=event_ts),
        )
        return self.idle_worker_metrics()

    def _record_worker_event_state(
        self,
        event: RolloutWorkerEvent,
        worker_id: str,
        event_ts: float,
    ) -> None:
        worker_id = str(worker_id)
        state = self._idle_workers.get(worker_id)
        if state is None:
            state = _IdleWorkerState(
                worker_id=worker_id,
                replica_rank=event.replica_rank,
            )
            self._idle_workers[worker_id] = state
        state.replica_rank = event.replica_rank
        state.event_ts = event_ts
        if event.event_type is RolloutWorkerEventType.GENERATION_STARTED:
            state.status = "generating"
            state.memory_released = False
            state.must_be_ready_at = None
        elif event.event_type is RolloutWorkerEventType.WORKER_IDLE:
            state.status = "idle"
            state.memory_released = event.memory_released
            state.must_be_ready_at = event.must_be_ready_at
        elif event.event_type is RolloutWorkerEventType.WORKER_RECLAIM_REQUESTED:
            state.status = "reclaiming"
        elif event.event_type is RolloutWorkerEventType.WORKER_READY:
            state.status = "ready"
            state.memory_released = False
            state.must_be_ready_at = None

    def register_idle_training_resource_metadata(
        self,
        metadata: Any,
    ) -> dict[str, float | int]:
        """Register true drafter training groups discovered from workers.

        Metadata is intentionally authoritative over ``group_size``.  If the
        drafter mesh uses both SP and DP collectives, workers report the whole
        connected mesh as ``full_collective_ranks``; scheduler then waits until
        every rank in that real group is idle.
        """

        records = _flatten_metadata_records(metadata)
        replica_group_members: dict[int, set[str]] = {}
        full_groups: list[tuple[str, ...]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for record in records:
            if not bool(record.get("in_drafter_train_group", False)):
                continue
            replica_rank = record.get("replica_rank")
            training_ranks = _normalize_worker_id_group(
                record.get("training_group_ranks", ())
            )
            if replica_rank is not None and training_ranks:
                replica_group_members.setdefault(int(replica_rank), set()).update(
                    training_ranks
                )
            full_group = _normalize_worker_id_group(
                record.get("full_collective_ranks", ())
            )
            if not full_group:
                full_group = training_ranks
            if full_group and full_group not in seen_groups:
                full_groups.append(full_group)
                seen_groups.add(full_group)
        replica_groups = {
            replica_rank: tuple(sorted(worker_ids, key=_natural_worker_sort_key))
            for replica_rank, worker_ids in sorted(replica_group_members.items())
        }
        self._replica_idle_worker_groups = replica_groups
        self._metadata_idle_training_groups = tuple(full_groups)
        metadata_summary = [
            {
                "rank": record.get("rank"),
                "worker_id": record.get("worker_id"),
                "replica_rank": record.get("replica_rank"),
                "in_group": bool(record.get("in_drafter_train_group", False)),
                "training_group_ranks": record.get("training_group_ranks", ()),
                "full_collective_ranks": record.get("full_collective_ranks", ()),
                "reason": record.get("reason", ""),
            }
            for record in records
        ]
        logger.warning(
            "[BubbleTime] resource_metadata groups=%s replica_groups=%s records=%s "
            "record_summary=%s",
            self._metadata_idle_training_groups,
            self._replica_idle_worker_groups,
            len(records),
            metadata_summary,
        )
        return {
            "bubble/registered_training_groups": len(full_groups),
            "bubble/registered_training_workers": len(
                {worker_id for group in full_groups for worker_id in group}
            ),
            "bubble/registered_replica_groups": len(replica_groups),
        }

    def idle_worker_metrics(self) -> dict[str, float | int]:
        return {
            "bubble/idle_workers": sum(
                int(state.status == "idle") for state in self._idle_workers.values()
            )
        }

    def rollout_idle_replica_ranks(self) -> tuple[int, ...]:
        """Return rollout replicas with registered drafter training resources."""

        return tuple(sorted(self._replica_idle_worker_groups))

    def rollout_idle_worker_ids_for_replica(
        self,
        replica_rank: int,
        *,
        fallback_worker_id: str | None = None,
    ) -> tuple[str, ...]:
        """Resolve drafter worker ids made idle by one rollout replica."""

        worker_ids = self._replica_idle_worker_groups.get(int(replica_rank), ())
        if worker_ids:
            return worker_ids
        if fallback_worker_id is not None:
            return (str(fallback_worker_id),)
        return (str(replica_rank),)

    def _idle_training_groups(
        self,
        config: DrafterScheduleConfig,
    ) -> tuple[tuple[str, ...], ...]:
        """Resolve legal collective groups for rollout-idle training.

        Explicit ``training_groups`` remains the most precise option.  In the
        common case, ``group_mode: auto`` builds stable groups from all known
        rollout workers, including workers that are currently still generating.
        That makes a half-idle collective group report ``incomplete`` instead
        of accidentally training only the idle subset.
        """

        if config.idle_worker_training_groups:
            return config.idle_worker_training_groups
        if self._metadata_idle_training_groups:
            return self._metadata_idle_training_groups
        if config.idle_worker_group_mode != "auto":
            return ()
        known_workers = tuple(sorted(self._idle_workers, key=_natural_worker_sort_key))
        if not known_workers:
            return ()
        if config.idle_worker_group_size is None:
            return ()
        group_size = config.idle_worker_group_size
        if group_size <= 1:
            return tuple((worker_id,) for worker_id in known_workers)
        groups: list[tuple[str, ...]] = []
        for start in range(0, len(known_workers), group_size):
            group = known_workers[start : start + group_size]
            if len(group) == group_size:
                groups.append(tuple(group))
        return tuple(groups)

    def select_idle_training_resources(
        self,
        config: DrafterScheduleConfig,
        *,
        now: float | None = None,
    ) -> AvailableTrainingResources:
        now = time.time() if now is None else now
        idle_states = {
            worker_id: state
            for worker_id, state in self._idle_workers.items()
            if state.status == "idle"
            and (
                not config.idle_worker_require_memory_released or state.memory_released
            )
        }
        groups = self._idle_training_groups(config)
        if not groups:
            if (
                config.idle_worker_group_mode == "auto"
                and not config.idle_worker_training_groups
                and not self._metadata_idle_training_groups
                and config.idle_worker_group_size is None
                and self._idle_workers
            ):
                logger.warning(
                    "[BubbleTime] idle_resource_skip reason=missing_training_group_metadata "
                    "group_mode=%s group_size=%s known_state=%s",
                    config.idle_worker_group_mode,
                    config.idle_worker_group_size,
                    _idle_state_summary(self._idle_workers, now=now),
                )
                return AvailableTrainingResources(
                    False, "missing_training_group_metadata"
                )
            logger.warning(
                "[BubbleTime] idle_resource_skip reason=no_idle_worker "
                "groups=%s known_state=%s require_memory_released=%s",
                groups,
                _idle_state_summary(self._idle_workers, now=now),
                config.idle_worker_require_memory_released,
            )
            return AvailableTrainingResources(False, "no_idle_worker")
        incomplete_seen = False
        for index, group in enumerate(groups):
            group = tuple(str(worker_id) for worker_id in group)
            missing = [worker_id for worker_id in group if worker_id not in idle_states]
            if missing:
                incomplete_seen = True
                logger.warning(
                    "[BubbleTime] idle_group_incomplete group_id=idle-group-%s "
                    "group=%s missing=%s idle_workers=%s known_state=%s",
                    index,
                    group,
                    missing,
                    tuple(sorted(idle_states, key=_natural_worker_sort_key)),
                    _idle_state_summary(self._idle_workers, now=now),
                )
                continue
            windows = [
                max(float(state.must_be_ready_at) - now, 0.0)
                for worker_id in group
                if (state := idle_states[worker_id]).must_be_ready_at is not None
            ]
            minimum_window = min(windows, default=math.inf)
            if math.isinf(minimum_window):
                minimum_window = self._effective_idle_min_window_sec(config)
            historical_window = self._effective_historical_idle_window_sec()
            if historical_window is not None:
                historical_remaining = min(
                    max(
                        historical_window - max(now - idle_states[worker_id].event_ts, 0.0),
                        0.0,
                    )
                    for worker_id in group
                )
                minimum_window = min(minimum_window, historical_remaining)
            min_idle_window_sec = self._effective_idle_min_window_sec(config)
            if minimum_window < min_idle_window_sec:
                logger.warning(
                    "[BubbleTime] idle_resource_skip reason=window_too_small "
                    "group_id=idle-group-%s group=%s minimum_window_s=%.3f "
                    "min_required_s=%.3f historical_window_s=%s now=%.3f "
                    "idle_state=%s",
                    index,
                    group,
                    minimum_window,
                    min_idle_window_sec,
                    historical_window,
                    now,
                    _idle_state_summary(self._idle_workers, now=now),
                )
                return AvailableTrainingResources(
                    False,
                    "window_too_small",
                    training_group_id=f"idle-group-{index}",
                    worker_ids=group,
                    minimum_idle_window_sec=minimum_window,
                )
            logger.warning(
                "[BubbleTime] idle_resource_ready group_id=idle-group-%s group=%s "
                "minimum_window_s=%.3f min_required_s=%.3f "
                "historical_window_s=%s now=%.3f",
                index,
                group,
                minimum_window,
                min_idle_window_sec,
                historical_window,
                now,
            )
            return AvailableTrainingResources(
                True,
                "training_group_ready",
                training_group_id=f"idle-group-{index}",
                worker_ids=group,
                minimum_idle_window_sec=minimum_window,
            )
        logger.warning(
            "[BubbleTime] idle_resource_skip reason=%s groups=%s idle_workers=%s "
            "known_state=%s",
            "incomplete_training_group" if incomplete_seen else "no_idle_worker",
            groups,
            tuple(sorted(idle_states, key=_natural_worker_sort_key)),
            _idle_state_summary(self._idle_workers, now=now),
        )
        return AvailableTrainingResources(
            False,
            "incomplete_training_group" if incomplete_seen else "no_idle_worker",
        )

    def prepare_training_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        allow_sync_fallback: bool = True,
    ) -> TrainingPlan:
        """Build a plan while avoiding worker RPCs for cheap skip conditions."""

        interval_matched = self.training_interval_matched(context.global_step, config)
        if config.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            return self.prepare_idle_worker_training_plan(
                context,
                config,
                allow_sync_fallback=allow_sync_fallback,
            )
        if (
            context.training_mode == "collect_only"
            or context.pending_training_count > 0
            or not interval_matched
        ):
            return self.plan_training(context, config)
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step, config=config
        )
        return self.plan_training(
            DrafterScheduleContext(
                global_step=context.global_step,
                training_mode=context.training_mode,
                collected_samples_this_step=context.collected_samples_this_step,
                oldlogprob_collection_requested=context.oldlogprob_collection_requested,
                data_status=data_status,
                pending_training_count=context.pending_training_count,
            ),
            config,
        )

    def prepare_idle_worker_training_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        *,
        allow_sync_fallback: bool = True,
    ) -> TrainingPlan:
        resources = self.select_idle_training_resources(config)
        if not resources.available:
            if allow_sync_fallback and self._should_sync_fallback(
                context.global_step, config
            ):
                return self._plan_sync_fallback_training(context, config)
            return self._skip_idle_worker_plan(context, config, resources)
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step,
            config=config,
            worker_ids=resources.worker_ids,
        )
        logger.info(
            "[BubbleTime] idle_data_status step=%s group=%s workers=%s "
            "trainable_batches=%s trainable_samples=%s buffer_samples=%s "
            "data_version=%s target_version=%s require_full_batch=%s "
            "sample_last_n_steps=%s",
            context.global_step,
            resources.training_group_id,
            resources.worker_ids,
            data_status.trainable_batches,
            data_status.trainable_samples,
            data_status.buffer_samples,
            data_status.data_version,
            data_status.target_version,
            config.require_full_batch,
            config.sample_last_n_steps,
        )
        return self.plan_training(
            DrafterScheduleContext(
                global_step=context.global_step,
                training_mode=context.training_mode,
                collected_samples_this_step=context.collected_samples_this_step,
                oldlogprob_collection_requested=context.oldlogprob_collection_requested,
                data_status=data_status,
                pending_training_count=context.pending_training_count,
            ),
            config,
            resources=resources,
            require_interval=False,
        )

    def _steps_without_training(self, global_step: object) -> int:
        try:
            step = _as_int(global_step)
        except (TypeError, ValueError):
            return 0
        if self._last_successful_training_step is None:
            return max(step, 0)
        return max(step - self._last_successful_training_step, 0)

    def _seconds_without_training(self) -> float:
        last_ts = self._last_successful_training_ts or self._training_progress_start_ts
        return max(time.time() - last_ts, 0.0)

    def _should_sync_fallback(
        self,
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        if not config.idle_worker_fallback_to_sync:
            return False
        max_steps = config.max_steps_without_training
        if max_steps is not None and self._steps_without_training(global_step) >= max(
            int(max_steps), 1
        ):
            return True
        max_seconds = config.idle_worker_max_seconds_without_training
        if max_seconds is not None and self._seconds_without_training() >= max(
            float(max_seconds), 0.0
        ):
            return True
        return False

    def _plan_sync_fallback_training(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingPlan:
        data_status = context.data_status or self.inspect_training_data(
            global_step=context.global_step,
            config=config,
            worker_ids=None,
        )
        fallback_context = DrafterScheduleContext(
            global_step=context.global_step,
            training_mode=context.training_mode,
            collected_samples_this_step=context.collected_samples_this_step,
            oldlogprob_collection_requested=context.oldlogprob_collection_requested,
            data_status=data_status,
            pending_training_count=context.pending_training_count,
        )
        plan = self.plan_training(
            fallback_context,
            replace(config, training_interval_steps=1),
        )
        fallback_reason = (
            "sync_fallback_training_ready"
            if plan.launch
            else (
                "sync_fallback_no_trainable_batch"
                if plan.reason == "no_trainable_batch"
                else plan.reason
            )
        )
        return replace(
            plan,
            reason=fallback_reason,
            execution_strategy=DrafterExecutionStrategy.SYNC,
            deadline_ts=None,
            target_worker_ids=(),
            training_group_id="sync-fallback",
        )

    def _skip_idle_worker_plan(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
        resources: AvailableTrainingResources,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        usable_window = max(
            resources.minimum_idle_window_sec
            - self._effective_idle_deadline_guard_sec(config)
            - self._effective_idle_startup_reserve_sec(config),
            0.0,
        )
        batch_estimate = self._effective_idle_batch_estimate_sec(config)
        return TrainingPlan(
            launch=False,
            reason=resources.reason,
            interval_matched=interval_matched,
            execution_strategy=DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER,
            source_global_step=context.global_step,
            max_batches=0,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            publish_after_success=False,
            required_target_version=(
                None
                if config.use_logits
                else (
                    context.data_status.target_version
                    if context.data_status is not None
                    else _as_int(context.global_step)
                )
            ),
            plan_id=uuid4().hex,
            target_worker_ids=resources.worker_ids,
            training_group_id=resources.training_group_id,
            idle_window_sec=resources.minimum_idle_window_sec,
            idle_usable_window_sec=usable_window,
            idle_window_batches=int(math.floor(usable_window / batch_estimate)),
            idle_batch_estimate_sec=batch_estimate,
        )

    def prepare_training_execution(self, plan: TrainingPlan) -> dict[str, Any]:
        if not plan.launch:
            return {"drafter/target_lm_head_synced": 0}
        if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            # The exact head for buffered step N data is cached before the step
            # N actor update.  Fetching the live actor head inside step N+1
            # rollout would both block the bubble and select the wrong version.
            return {"drafter/target_lm_head_synced": 0}
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        return self._worker_executor.prepare_training(plan)

    def activate_training_workers(self) -> list[Any]:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        results = self._worker_executor.activate_training_workers()
        active = [
            result
            for result in results
            if isinstance(result, dict)
            and result.get("reason") not in {"disabled", "not_in_training_group"}
        ]
        failures = [result for result in active if not result.get("activated", False)]
        if failures:
            raise RuntimeError(
                f"SPECO drafter trainer activation failed: {failures[:3]}"
            )
        return results

    def wait_pending_publish(self) -> int:
        if self._publish_executor is None:
            return 0
        return self._publish_executor.wait_pending()

    @staticmethod
    def should_collect(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        return step_matches_interval(global_step, config.collect_interval_steps)

    def plan_collection(
        self,
        context: DrafterCollectionContext,
        config: DrafterScheduleConfig,
    ) -> CollectionPlan:
        collect_interval_matched = self.should_collect(context.global_step, config)
        training_interval_matched = self.training_interval_matched(
            context.global_step, config
        )
        common: Any = {
            "collection_id": uuid4().hex,
            "source": context.source,
            "source_global_step": context.global_step,
            "collect_interval_matched": collect_interval_matched,
            "training_interval_matched": training_interval_matched,
            "sample_rate": config.collection_sample_rate,
            "max_samples_per_replica": config.max_collect_samples_per_replica,
            "max_tokens_per_replica": config.max_collect_tokens_per_replica,
            "hidden_window_mode": config.hidden_window_mode,
            "hidden_window_tokens_per_sample": config.hidden_window_tokens_per_sample,
            "hidden_window_min_rows": config.hidden_window_min_rows,
        }
        if not context.drafter_enabled:
            return CollectionPlan(collect=False, reason="drafter_disabled", **common)
        if not context.source_enabled:
            return CollectionPlan(collect=False, reason="source_disabled", **common)
        if context.validation:
            return CollectionPlan(collect=False, reason="validation", **common)
        if not collect_interval_matched:
            return CollectionPlan(
                collect=False, reason="interval_not_reached", **common
            )
        if context.require_training_interval and not training_interval_matched:
            return CollectionPlan(
                collect=False,
                reason="training_interval_not_reached",
                **common,
            )
        if config.collection_sample_rate <= 0:
            return CollectionPlan(collect=False, reason="sample_rate_zero", **common)
        return CollectionPlan(collect=True, reason="collection_enabled", **common)

    def execute_collection_plan(self, plan: CollectionPlan, payload: CollectionPayload):
        if self._collection_executor is None:
            raise RuntimeError("Drafter collection executor has not been bound")
        return self.collection_strategy.execute(
            plan,
            payload,
            executor=self._collection_executor,
        )

    def on_collection_ready(self, plan: CollectionPlan, payload: CollectionPayload):
        """Lifecycle Facade event for a source adapter's prepared payload."""
        return self.execute_collection_plan(plan, payload)

    def on_before_actor_update(
        self, context: BeforeActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Plan and prepare drafter training before the PPO actor update."""
        plan = self.prepare_training_plan(
            context.schedule_context,
            context.config,
            allow_sync_fallback=context.allow_sync_fallback,
        )
        metrics: dict[str, Any] = dict(plan.metrics())
        metrics.update(
            {
                "scheduler/train_requested": int(plan.launch),
                "scheduler/planned_batches": int(plan.max_batches),
                "bubble/starvation_steps": self._steps_without_training(
                    context.schedule_context.global_step
                ),
                "bubble/starvation_seconds": self._seconds_without_training(),
                "bubble/sync_fallback_requested": int(
                    plan.reason.startswith("sync_fallback")
                ),
                "bubble/sync_fallback_launched": int(
                    plan.launch and plan.reason == "sync_fallback_training_ready"
                ),
                "bubble/sync_fallback_batches": (
                    int(plan.max_batches)
                    if plan.launch and plan.reason == "sync_fallback_training_ready"
                    else 0
                ),
            }
        )
        metrics.update(self.prepare_training_execution(plan))
        return SchedulerEventOutcome(
            training_plan=plan,
            metrics=metrics,
        )

    def on_after_actor_update(
        self, context: AfterActorUpdateContext
    ) -> SchedulerEventOutcome:
        """Execute the prepared plan after the PPO actor update."""
        plan = context.training_plan
        if not plan.launch:
            return SchedulerEventOutcome(training_plan=plan, metrics={})
        execution = self.execute_training_plan(
            plan,
            runtime_state=context.runtime_state,
        )
        if execution.reason == "submitted_async":
            metrics = {
                "scheduler/train_launched": 1,
                "scheduler/pending_training_count": 1,
            }
            metrics.update(context.runtime_state.metrics())
            return SchedulerEventOutcome(
                training_plan=plan,
                metrics=metrics,
            )
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=context.runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return SchedulerEventOutcome(
            training_plan=plan,
            training_execution=outcome,
            metrics=outcome.metrics,
        )

    def on_safe_point(self, context: AfterWeightUpdateContext) -> SchedulerEventOutcome:
        """Plan and execute publication at a rollout-safe lifecycle point."""
        plan = self.plan_publish(
            global_step=context.global_step,
            drafter_trained=context.drafter_trained,
            config=context.config,
            training_plan=context.training_plan,
        )
        outcome = self.execute_publish_plan(plan)
        return SchedulerEventOutcome(
            training_plan=context.training_plan,
            publish_plan=plan,
            publish_outcome=outcome,
            metrics=outcome.metrics(),
        )

    def on_after_weight_update(
        self, context: AfterWeightUpdateContext
    ) -> SchedulerEventOutcome:
        """Named lifecycle alias for the post-weight-update safe point."""
        return self.on_safe_point(context)

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
        resources: AvailableTrainingResources | None = None,
        *,
        require_interval: bool = True,
    ) -> TrainingPlan:
        interval_matched = self.training_interval_matched(context.global_step, config)
        execution_strategy = (
            DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
            if resources is not None
            else DrafterExecutionStrategy.SYNC
        )
        trigger = self.trigger_policy.should_train(
            context,
            config,
            interval_matched=interval_matched if require_interval else True,
        )
        budget = self.sync_budget_policy.make_budget(context, config)
        if resources is not None and budget.max_batches > 0:
            deadline_guard_sec = self._effective_idle_deadline_guard_sec(config)
            startup_reserve_sec = self._effective_idle_startup_reserve_sec(config)
            batch_estimate = self._effective_idle_batch_estimate_sec(config)
            max_batches_per_window = self._effective_idle_max_batches_per_window(config)
            usable_window = max(
                resources.minimum_idle_window_sec
                - deadline_guard_sec
                - startup_reserve_sec,
                0.0,
            )
            window_batches = int(math.floor(usable_window / batch_estimate))
            max_batches = min(
                window_batches,
                context.data_status.trainable_batches if context.data_status else 0,
                max_batches_per_window,
                budget.max_batches,
            )
            trainable_batches = (
                context.data_status.trainable_batches if context.data_status else 0
            )
            if max_batches > 0:
                idle_budget_reason = "idle_worker_budget_ready"
            elif window_batches <= 0:
                idle_budget_reason = "window_too_small"
            elif trainable_batches <= 0:
                idle_budget_reason = "no_trainable_batch"
            elif max_batches_per_window <= 0 or budget.max_batches <= 0:
                idle_budget_reason = "no_training_budget"
            else:
                idle_budget_reason = "no_training_budget"
            budget = TrainingBudget(
                max_batches=max_batches,
                min_batches=budget.min_batches,
                deadline_ts=time.time() + usable_window,
                require_full_batch=budget.require_full_batch,
                sample_last_n_steps=budget.sample_last_n_steps,
                reason=idle_budget_reason,
            )
            logger.info(
                "[BubbleTime] idle_budget step=%s group=%s workers=%s "
                "minimum_window_s=%.3f usable_window_s=%.3f guard_s=%.3f "
                "startup_reserve_s=%.3f "
                "batch_estimate_s=%.3f window_batches=%s trainable_batches=%s "
                "max_batches_per_window=%s sync_budget_batches=%s "
                "planned_batches=%s reason=%s estimate_source=%s",
                context.global_step,
                resources.training_group_id,
                resources.worker_ids,
                resources.minimum_idle_window_sec,
                usable_window,
                deadline_guard_sec,
                startup_reserve_sec,
                batch_estimate,
                window_batches,
                context.data_status.trainable_batches if context.data_status else 0,
                max_batches_per_window,
                self.sync_budget_policy.make_budget(context, config).max_batches,
                max_batches,
                budget.reason,
                (
                    "bootstrap"
                    if self._idle_batch_estimate_is_bootstrap(config)
                    else (
                        "config"
                        if config.idle_worker_initial_batch_estimate_sec is not None
                        else "history"
                    )
                ),
            )
        common: Any = {
            "interval_matched": interval_matched,
            "execution_strategy": execution_strategy,
            "source_global_step": context.global_step,
            "max_batches": budget.max_batches,
            "min_batches": budget.min_batches,
            "deadline_ts": budget.deadline_ts,
            "require_full_batch": budget.require_full_batch,
            "sample_last_n_steps": budget.sample_last_n_steps,
            "data_version": (
                context.data_status.data_version if context.data_status else None
            ),
            "required_target_version": (
                None
                if config.use_logits
                else (
                    context.data_status.target_version
                    if context.data_status is not None
                    else _as_int(context.global_step)
                )
            ),
            "plan_id": uuid4().hex,
            "worker_snapshots": (
                context.data_status.worker_snapshots if context.data_status else None
            ),
            "target_worker_ids": resources.worker_ids if resources else (),
            "training_group_id": resources.training_group_id if resources else "",
            "idle_window_sec": (
                resources.minimum_idle_window_sec if resources is not None else None
            ),
            "idle_usable_window_sec": (
                max(
                    resources.minimum_idle_window_sec
                    - self._effective_idle_deadline_guard_sec(config)
                    - self._effective_idle_startup_reserve_sec(config),
                    0.0,
                )
                if resources is not None
                else None
            ),
            "idle_window_batches": (
                int(
                    math.floor(
                        max(
                            resources.minimum_idle_window_sec
                            - self._effective_idle_deadline_guard_sec(config)
                            - self._effective_idle_startup_reserve_sec(config),
                            0.0,
                        )
                        / self._effective_idle_batch_estimate_sec(config)
                    )
                )
                if resources is not None
                else None
            ),
            "idle_batch_estimate_sec": (
                self._effective_idle_batch_estimate_sec(config)
                if resources is not None
                else None
            ),
            "idle_startup_reserve_sec": (
                self._effective_idle_startup_reserve_sec(config)
                if resources is not None
                else None
            ),
            "idle_trainable_batches": (
                context.data_status.trainable_batches
                if resources is not None and context.data_status is not None
                else None
            ),
        }
        if not trigger.should_train:
            return TrainingPlan(
                launch=False,
                reason=trigger.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches <= 0:
            return TrainingPlan(
                launch=False,
                reason=budget.reason,
                publish_after_success=False,
                **common,
            )
        if budget.max_batches < budget.min_batches:
            return TrainingPlan(
                launch=False,
                reason="insufficient_training_budget",
                publish_after_success=False,
                **common,
            )
        return TrainingPlan(
            launch=True,
            reason=trigger.reason,
            publish_after_success=self._publish_interval_matched(
                context.global_step, config
            ),
            **common,
        )

    def execute_training_plan(self, plan: TrainingPlan, *, runtime_state):
        """Execute through the strategy selected by the generated plan."""

        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")

        if plan.execution_strategy is DrafterExecutionStrategy.SYNC:
            return self.sync_execution_strategy.execute(
                plan,
                executor=self._worker_executor,
                runtime_state=runtime_state,
            )
        if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            return self.rollout_idle_execution_strategy.execute(
                plan,
                executor=self._worker_executor,
                runtime_state=runtime_state,
            )
        raise NotImplementedError(
            f"Unsupported drafter execution strategy: {plan.execution_strategy.value}"
        )

    def poll_pending_training(self, *, runtime_state) -> TrainingOutcome | None:
        plan = runtime_state.active_plan
        if plan is None:
            return None
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        execution = self.rollout_idle_execution_strategy.poll(
            executor=self._worker_executor,
            runtime_state=runtime_state,
        )
        if execution is None:
            return None
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return outcome

    def wait_pending_training(self, *, runtime_state) -> TrainingOutcome | None:
        plan = runtime_state.active_plan
        if plan is None:
            return None
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        execution = self.rollout_idle_execution_strategy.wait(
            executor=self._worker_executor,
            runtime_state=runtime_state,
        )
        if execution is None:
            return None
        outcome = TrainingOutcome.from_execution(
            execution,
            runtime_state=runtime_state,
            plan=plan,
        )
        self._record_training_outcome(plan, outcome)
        return outcome

    def _record_training_outcome(
        self,
        plan: TrainingPlan,
        outcome: TrainingOutcome,
    ) -> None:
        if plan.execution_strategy is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER:
            self.record_idle_training_outcome(outcome)
        if outcome.trained and outcome.successful_steps > 0:
            try:
                self._last_successful_training_step = _as_int(plan.source_global_step)
            except (TypeError, ValueError):
                self._last_successful_training_step = None
            self._last_successful_training_ts = time.time()
            worker_batch_estimates = [
                max(float(result.elapsed_sec), 0.0) / max(result.successful_steps, 1)
                for result in outcome.worker_results
                if result.successful_steps > 0 and result.elapsed_sec > 0
            ]
            batch_sec = max(
                worker_batch_estimates,
                default=max(float(outcome.elapsed_sec), 0.0)
                / max(int(outcome.successful_steps), 1),
            )
            if batch_sec > 0:
                self._idle_worker_batch_samples_sec.append(batch_sec)
                if self._idle_worker_batch_estimate_sec is None:
                    self._idle_worker_batch_estimate_sec = batch_sec
                else:
                    self._idle_worker_batch_estimate_sec = (
                        self._idle_worker_batch_estimate_sec * 0.8 + batch_sec * 0.2
                    )
                logger.info(
                    "[BubbleTime] updated idle batch estimate: step=%s "
                    "observed_batch_s=%.3f estimated_batch_s=%.3f "
                    "successful_steps=%s elapsed_s=%.3f",
                    plan.source_global_step,
                    batch_sec,
                    self._idle_worker_batch_estimate_sec,
                    outcome.successful_steps,
                    outcome.elapsed_sec,
                )

    def request_reclaim(self, worker_ids: tuple[str, ...]) -> Any:
        if self._worker_executor is None:
            raise RuntimeError("Drafter worker executor has not been bound")
        for worker_id in worker_ids:
            if worker_id in self._idle_workers:
                self._idle_workers[worker_id].status = "reclaiming"
        request = self._worker_executor.request_reclaim(
            tuple(str(worker_id) for worker_id in worker_ids)
        )
        return self._worker_executor.resolve(request)

    @staticmethod
    def _publish_interval_matched(
        global_step: object,
        config: DrafterScheduleConfig,
    ) -> bool:
        interval = _as_int(config.publish_interval_steps or 0)
        return interval <= 0 or _as_int(global_step) % interval == 0

    @staticmethod
    def plan_publish(
        *,
        global_step: object,
        drafter_trained: bool,
        config: DrafterScheduleConfig,
        training_plan: TrainingPlan | None = None,
    ) -> PublishPlan:
        # A Bubble Time publish is issued only after the upstream actor weight
        # update has resumed vLLM.  Keep the transfer off that critical path;
        # the generation hook waits only if it is still pending at its next
        # safe point.  The legacy synchronous path continues to honor the
        # explicit publish_async setting.
        asynchronous = config.publish_async or (
            training_plan is not None
            and training_plan.execution_strategy
            is DrafterExecutionStrategy.ROLLOUT_IDLE_WORKER
        )
        if not drafter_trained or (
            training_plan is not None and not training_plan.publish_after_success
        ):
            return PublishPlan(
                publish=False,
                reason=(
                    "drafter_not_trained"
                    if not drafter_trained
                    else "training_plan_publish_disabled"
                ),
                interval_matched=False,
                source_global_step=global_step,
                asynchronous=asynchronous,
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
            asynchronous=asynchronous,
        )

    def execute_publish_plan(self, plan: PublishPlan):
        if self._publish_executor is None:
            raise RuntimeError("Drafter publish executor has not been bound")
        return self.publish_execution_strategy.execute(
            plan, executor=self._publish_executor
        )
