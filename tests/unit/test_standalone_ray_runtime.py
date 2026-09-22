# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import sys
from types import SimpleNamespace

import pytest

from verl_speco.standalone_ray_runtime import (
    StandaloneProducerActor,
    StandaloneRayTrainer,
    _configure_driver_logging,
)
from verl_speco.standalone_tq_training_launcher import PipelineCommands


def _commands() -> PipelineCommands:
    return PipelineCommands(
        vllm=None,
        vllm_endpoints=("http://127.0.0.1:8000/v1",),
        owner=["python", "-m", "verl_speco.tq_owner"],
        producer=["producer"],
        consumer=["consumer"],
        producer_overrides=("producer=true",),
        consumer_overrides=("consumer=true",),
    )


def test_producer_actor_reuses_existing_producer_and_supports_stop(monkeypatch) -> None:
    started = asyncio.Event()

    async def fake_run_producer(config, **kwargs):
        assert config == {"producer": True}
        assert set(kwargs) == {
            "before_request",
            "on_published",
            "get_runtime_state",
        }
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    actor = StandaloneProducerActor({"producer": True})

    async def exercise():
        assert await actor.start() == {"started": True}
        await started.wait()
        assert await actor.start() == {"started": False}
        assert await actor.stop() == {"stopped": True}

    asyncio.run(exercise())


def test_producer_actor_coalesces_publish_notifications_as_cumulative_total(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class _Stats:
        published_count: int

    class _Events:
        def __init__(self):
            self.items = []

        def put(self, value):
            self.items.append(value)

    async def fake_run_producer(config, **kwargs):
        for sequence_no in range(3):
            await kwargs["on_published"](sequence_no)
        return _Stats(published_count=3)

    monkeypatch.setitem(
        sys.modules,
        "verl_speco.standalone_tq_producer",
        SimpleNamespace(run_producer=fake_run_producer),
    )
    events = _Events()
    actor = StandaloneProducerActor({"producer": True}, events)

    async def exercise():
        assert await actor.start() == {"started": True}
        assert await actor.wait() == {"published_count": 3}

    asyncio.run(exercise())

    assert events.items == [
        {"kind": "samples_published", "published_total": 3}
    ]


@dataclass(frozen=True)
class _Ref:
    name: str


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self):
        return self.callback()


class _ProducerHandle:
    def __init__(self):
        self.stopped = False
        self.start = _RemoteMethod(lambda: _Ref("start"))
        self.wait = _RemoteMethod(lambda: _Ref("producer"))
        self.stop = _RemoteMethod(self._stop)

    def _stop(self):
        self.stopped = True
        return _Ref("stop")


class _RemoteProducerClass:
    def __init__(self, handle):
        self.handle = handle

    def options(self, **kwargs):
        assert kwargs["name"] == "speco_standalone_producer"
        return self

    def remote(self, config, events):
        assert config is _runtime_config()
        return self.handle


class _WorkerGroup:
    def run_standalone_training(self):
        return [_Ref("consumer-0"), _Ref("consumer-1")]


class _FakeRay:
    def __init__(self, producer):
        self.producer = producer

    def remote(self, **kwargs):
        assert kwargs == {"max_concurrency": 4}
        return lambda cls: _RemoteProducerClass(self.producer)

    def get(self, ref):
        return {"ref": ref.name}

    def wait(self, refs, num_returns, timeout=0):
        assert num_returns == 1
        consumer = next(ref for ref in refs if ref.name.startswith("consumer"))
        return [consumer], [ref for ref in refs if ref != consumer]


class _Queue:
    def put(self, value):
        pass

    def get(self, *, block, timeout):
        raise AssertionError("event queue should not be read in this test")


class _Store:
    def connect(self):
        pass

    def list_ready(self):
        return []

    def close(self):
        pass


_CONFIG = None


def _runtime_config():
    global _CONFIG
    if _CONFIG is None:
        training = SimpleNamespace(batch_size_per_gpu=4, transfer_queue={})
        training.get = lambda key, default=None: getattr(training, key, default)
        draft_training = SimpleNamespace(
            nproc_per_node=1,
            nnodes=1,
            scheduler={},
        )
        draft_training.get = lambda key, default=None: getattr(
            draft_training, key, default
        )
        _CONFIG = SimpleNamespace(
            speco=SimpleNamespace(
                draft_training=draft_training,
                standalone_tq_producer=SimpleNamespace(max_pending_samples=20),
            ),
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    drafter=SimpleNamespace(training=training)
                )
            ),
        )
    return _CONFIG


def test_queue_config_defaults_low_watermark_to_half_producer_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(_commands(), ray_module=SimpleNamespace())

    config = trainer._queue_config()

    assert config.global_batch_size == 4
    assert config.low_watermark_samples == 10
    assert config.high_watermark_samples == 20


def test_driver_logging_overrides_preinstalled_warning_handler(monkeypatch) -> None:
    monkeypatch.delenv("SPECO_STANDALONE_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]
        root.setLevel(logging.WARNING)

        _configure_driver_logging()

        assert root.level == logging.INFO
        assert handler.level == logging.INFO
        assert (
            logging.getLogger("verl_speco.standalone_ray_runtime").level
            == logging.INFO
        )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_driver_logging_supports_scheduler_debug(monkeypatch) -> None:
    monkeypatch.setenv("SPECO_STANDALONE_LOG_LEVEL", "DEBUG")
    root = logging.getLogger()
    handler = logging.StreamHandler()
    original_handlers = list(root.handlers)
    original_level = root.level
    module_logger = logging.getLogger("verl_speco.standalone_ray_runtime")
    original_module_level = module_logger.level
    try:
        root.handlers[:] = [handler]

        _configure_driver_logging()

        assert root.level == logging.DEBUG
        assert handler.level == logging.DEBUG
        assert module_logger.level == logging.DEBUG
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        module_logger.setLevel(original_module_level)


def test_trainer_stops_producer_after_all_consumer_workers_finish(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FakeRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    assert trainer.run() == 0
    assert producer.stopped is True


class _FailingRay(_FakeRay):
    def get(self, ref):
        if ref.name == "consumer-0":
            raise RuntimeError("consumer failed")
        return super().get(ref)


def test_trainer_propagates_consumer_error_and_stops_producer(monkeypatch) -> None:
    producer = _ProducerHandle()
    ray = _FailingRay(producer)
    monkeypatch.setattr(
        "verl_speco.standalone_ray_runtime.compose_runtime_config",
        lambda overrides: _runtime_config(),
    )
    trainer = StandaloneRayTrainer(
        _commands(),
        ray_module=ray,
        worker_group_factory=lambda config, **kwargs: _WorkerGroup(),
        queue_factory=_Queue,
        feature_store_factory=lambda config: _Store(),
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        trainer.run()
    assert producer.stopped is True
