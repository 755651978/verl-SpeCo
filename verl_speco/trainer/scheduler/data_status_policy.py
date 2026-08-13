# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Policies for aggregating drafter-worker buffer availability."""

from __future__ import annotations

from verl_speco.trainer.scheduler.schedule_types import TrainingDataStatus


class ConservativeTrainingDataStatusPolicy:
    """Use capacity available on every rank in a distributed training group."""

    def aggregate(
        self, statuses: list[TrainingDataStatus], *, global_step: object
    ) -> TrainingDataStatus | None:
        if not statuses:
            return None
        target_version_consistent = all(
            s.target_version_consistent for s in statuses
        ) and all(s.target_version == statuses[0].target_version for s in statuses)
        newest_sample_step = max(
            (s.newest_sample_step for s in statuses if s.newest_sample_step is not None),
            default=None,
        )
        return TrainingDataStatus(
            current_step=int(global_step),
            current_step_samples=min(s.current_step_samples for s in statuses),
            buffer_samples=min(s.buffer_samples for s in statuses),
            trainable_samples=min(s.trainable_samples for s in statuses),
            trainable_batches=min(s.trainable_batches for s in statuses),
            batch_size_per_gpu=max(s.batch_size_per_gpu for s in statuses),
            partial_batch_available=all(s.partial_batch_available for s in statuses),
            oldest_sample_step=min(
                (s.oldest_sample_step for s in statuses if s.oldest_sample_step is not None),
                default=None,
            ),
            newest_sample_step=newest_sample_step,
            same_step_data_required=any(s.same_step_data_required for s in statuses),
            target_version=(statuses[0].target_version if target_version_consistent else None),
            target_version_consistent=target_version_consistent,
            data_version=newest_sample_step,
        )
