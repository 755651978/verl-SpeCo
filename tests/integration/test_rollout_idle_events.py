# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from types import SimpleNamespace

import verl_speco.integration.rollout_idle_events as rollout_idle_events


def test_drafter_sample_event_uses_object_store_reference(monkeypatch) -> None:
    emitted = []
    sample_ref = object()
    actor = SimpleNamespace(
        emit=SimpleNamespace(remote=lambda event: emitted.append(event))
    )
    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        get_actor=lambda name: actor,
        put=lambda sample: sample_ref,
    )
    monkeypatch.setattr(rollout_idle_events, "_ray_module", lambda: fake_ray)

    emitted_ok = rollout_idle_events.emit_rollout_drafter_sample(
        "bubble-bus",
        {"hidden_states": "large-payload"},
        sample_id="5:0:req-1",
        replica_rank=0,
        global_step=5,
    )

    assert emitted_ok is True
    assert emitted == [
        {
            "event_type": rollout_idle_events.DRAFTER_SAMPLE_READY_EVENT,
            "sample_id": "5:0:req-1",
            "sample_ref": sample_ref,
            "replica_rank": 0,
            "global_step": 5,
            "event_ts": emitted[0]["event_ts"],
        }
    ]
