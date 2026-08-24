# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Ray-backed event bus for fine-grained rollout idle-worker events."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPECO_ROLLOUT_IDLE_EVENT_BUS_ENV = "VERL_SPECO_ROLLOUT_IDLE_EVENT_BUS"


def _ray_module():
    try:
        import ray
    except Exception:  # noqa: BLE001
        return None
    return ray


def _event_bus_actor_class(ray):
    @ray.remote
    class RolloutIdleEventBus:
        def __init__(self) -> None:
            self._events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> int:
            event = dict(event)
            event.setdefault("event_ts", time.time())
            self._events.append(event)
            return len(self._events)

        def drain(self) -> list[dict[str, Any]]:
            events = self._events
            self._events = []
            return events

        def clear(self) -> int:
            count = len(self._events)
            self._events = []
            return count

    return RolloutIdleEventBus


def ensure_rollout_idle_event_bus(name: str):
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return None
    try:
        return ray.get_actor(name)
    except Exception:  # noqa: BLE001
        pass
    try:
        actor_cls = _event_bus_actor_class(ray)
        return actor_cls.options(name=name, lifetime="detached").remote()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to create SPECO rollout idle event bus %s: %s", name, exc)
        return None


def emit_rollout_idle_event(name: str | None, event: dict[str, Any]) -> bool:
    if not name:
        return False
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return False
    try:
        actor = ray.get_actor(name)
        actor.emit.remote(dict(event))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to emit SPECO rollout idle event to %s: %s", name, exc)
        return False


def drain_rollout_idle_events(name: str | None) -> list[dict[str, Any]]:
    if not name:
        return []
    ray = _ray_module()
    if ray is None or not getattr(ray, "is_initialized", lambda: False)():
        return []
    try:
        actor = ray.get_actor(name)
        return list(ray.get(actor.drain.remote()) or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to drain SPECO rollout idle events from %s: %s", name, exc)
        return []
