# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import pytest

from verl_speco.trainer.scheduler import (
    DrafterScheduler,
    DrafterTrainingDataSource,
    ProducerAction,
    QueueScheduleContext,
    QueueScheduleConfig,
    QueueStatus,
)


def _config() -> QueueScheduleConfig:
    return QueueScheduleConfig(
        low_watermark_samples=16,
        high_watermark_samples=32,
        global_batch_size=16,
    )


def _context(**status_kwargs) -> QueueScheduleContext:
    runtime_keys = {"producer_done", "producer_paused", "consumer_training"}
    runtime = {
        key: status_kwargs.pop(key)
        for key in tuple(status_kwargs)
        if key in runtime_keys
    }
    return QueueScheduleContext(
        queue_status=QueueStatus(**status_kwargs),
        config=_config(),
        **runtime,
    )


def test_transfer_queue_collection_applies_watermarks_and_hysteresis() -> None:
    scheduler = DrafterScheduler()

    high = scheduler.plan_queue_collection(_context(ready_samples=32))
    low = scheduler.plan_queue_collection(
        _context(ready_samples=16, producer_paused=True)
    )
    middle_paused = scheduler.plan_queue_collection(
        _context(ready_samples=20, producer_paused=True)
    )
    middle_running = scheduler.plan_queue_collection(
        _context(ready_samples=20, producer_paused=False)
    )

    assert not high.collect
    assert high.reason == "high_watermark_reached"
    assert high.producer_action is ProducerAction.PAUSE
    assert high.max_new_samples is None
    assert high.metrics()["drafter/collection_plan_source"] == 3
    assert high.metrics()["drafter/collection_plan_reason"] == 9
    assert low.collect
    assert low.reason == "low_watermark_reached"
    assert low.producer_action is ProducerAction.RUN
    assert low.max_new_samples is None
    assert middle_paused.producer_action is ProducerAction.PAUSE
    assert middle_paused.reason == "watermark_hysteresis_paused"
    assert middle_running.producer_action is ProducerAction.RUN
    assert middle_running.reason == "watermark_hysteresis_running"


def test_finished_transfer_queue_producer_stops_at_any_watermark() -> None:
    plan = DrafterScheduler().plan_queue_collection(
        _context(ready_samples=0, producer_done=True)
    )

    assert not plan.collect
    assert plan.reason == "producer_done"
    assert plan.producer_action is ProducerAction.STOP


def test_transfer_queue_training_does_not_submit_twice() -> None:
    plan = DrafterScheduler().plan_queue_training(
        _context(ready_samples=32, consumer_training=True)
    )

    assert not plan.launch
    assert plan.reason == "consumer_training"


def test_transfer_queue_training_requires_one_complete_global_batch() -> None:
    scheduler = DrafterScheduler()

    insufficient = scheduler.plan_queue_training(_context(ready_samples=15))
    ready = scheduler.plan_queue_training(
        _context(ready_samples=16), selected_keys=("sample-0", "sample-1")
    )

    assert not insufficient.launch
    assert insufficient.reason == "insufficient_ready_samples"
    assert ready.launch
    assert ready.max_batches == 1
    assert ready.require_full_batch
    assert ready.data_source is DrafterTrainingDataSource.TRANSFER_QUEUE
    assert ready.required_samples == 16
    assert ready.selected_keys == ("sample-0", "sample-1")
    assert ready.to_worker_payload()["selected_keys"] == ["sample-0", "sample-1"]
    assert ready.to_worker_payload()["data_source"] == "transfer_queue"
    assert ready.to_worker_payload()["required_samples"] == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "low_watermark_samples": 1,
            "high_watermark_samples": 1,
            "global_batch_size": 0,
        },
        {
            "low_watermark_samples": -1,
            "high_watermark_samples": 16,
            "global_batch_size": 16,
        },
        {
            "low_watermark_samples": 16,
            "high_watermark_samples": 16,
            "global_batch_size": 16,
        },
        {
            "low_watermark_samples": 8,
            "high_watermark_samples": 32,
            "global_batch_size": 16,
        },
    ],
)
def test_transfer_queue_schedule_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        QueueScheduleConfig(**kwargs)
