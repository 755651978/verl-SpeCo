# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Completion, capacity and cancellation contracts for computed features."""

import threading

import pytest
import torch

from verl_speco.producer.feature_export import (
    FeatureExportQueue,
    wait_for_exports,
    rollout_feature_fields,
)


def test_export_is_nonblocking_bounded_and_waits_for_completion(tmp_path):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue(max_bytes=32, max_pending=1)
    entered, release = threading.Event(), threading.Event()
    payload = {"hidden": torch.tensor([1.0, 2.0])}

    def wait():
        entered.set()
        assert release.wait(5)

    try:
        assert queue.reserve(32)
        queue.submit_reserved(tmp_path, payload, size=32, wait=wait)
        assert entered.wait(5)
        assert not list(tmp_path.glob("*.pt"))
        assert not queue.reserve(1)
        with pytest.raises(TimeoutError):
            wait_for_exports(tmp_path, timeout=0)
    finally:
        release.set()
        queue.close()
    wait_for_exports(tmp_path)
    data = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert torch.equal(data["hidden"], payload["hidden"])
    assert queue.reserve(32)
    queue.release(32)


def test_export_failure_is_visible_and_releases_capacity(tmp_path, monkeypatch):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue(max_bytes=16)

    def fail(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(torch, "save", fail)
    assert queue.reserve(16)
    queue.submit_reserved(tmp_path, {}, size=16)
    queue.close()
    assert (tmp_path / "error").exists()
    assert not list(tmp_path.glob("*.pending"))
    assert queue.reserve(16)
    queue.release(16)


def test_cancellation_before_completion_does_not_publish(tmp_path):
    (tmp_path / "live").touch()
    queue = FeatureExportQueue()
    gate = threading.Event()
    assert queue.reserve(16)
    queue.submit_reserved(tmp_path, {}, size=16, wait=lambda: gate.wait(5))
    (tmp_path / "live").unlink()
    gate.set()
    queue.close()
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.pending"))


def test_shared_fields_preserve_tokens_positions_and_storage():
    prompt, response = torch.tensor([1, 2]), torch.tensor([3, 4])
    hidden = torch.randn(2, 8)
    positions = torch.tensor([1, 2])
    sample = rollout_feature_fields(prompt, response, hidden, positions)
    assert sample["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert sample["hidden_states"].data_ptr() == hidden.data_ptr()
    assert sample["hidden_positions"].tolist() == [[1, 2]]
    with pytest.raises(ValueError, match="match feature rows"):
        rollout_feature_fields(prompt, response, hidden, positions[:1])


def test_failed_export_does_not_wait_for_unrelated_pending_chunk(tmp_path):
    (tmp_path / "other.pending").touch()
    (tmp_path / "error").touch()
    wait_for_exports(tmp_path, timeout=0)
