# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest
import requests
import torch

from verl_speco.integration import external_vllm_weight_sync as sync


@pytest.fixture
def sender(monkeypatch):
    calls = []
    groups = []
    sends = []

    class Session:
        trust_env = True

        def request(self, method, url, json, params, timeout):
            calls.append((method, url, json, params))
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"world_size": 2}
                if url.endswith("get_world_size")
                else {},
            )

        def close(self):
            calls.append(("CLOSE", "session", None, None))

    def trainer_init(info):
        group = SimpleNamespace(
            available=True, disabled=False, destroy=Mock(), info=info
        )
        groups.append(group)
        return group

    engine = SimpleNamespace(
        trainer_init=trainer_init,
        trainer_send_weights=lambda iterator, args: sends.append(
            (list(iterator), args)
        ),
    )
    ports = iter([29001, 29002])
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.weight_transfer.nccl_engine",
        SimpleNamespace(NCCLWeightTransferEngine=engine),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.network_utils",
        SimpleNamespace(get_ip=lambda: "10.0.0.1", get_open_port=lambda: next(ports)),
    )
    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    value = sync.ExternalVllmWeightSender(
        {
            "endpoints": ["http://one:8000/v1/", "http://two:8001/v1"],
            "bucket_size_mb": 1,
            "packed": True,
        }
    )
    # Test the real HTTP/transfer orchestration with CPU tensor stand-ins.
    monkeypatch.setattr(
        value, "_batches", lambda iterator: iter([[item] for item in iterator])
    )
    return value, calls, groups, sends


def test_init_uses_one_nccl_group_per_endpoint(sender):
    value, calls, groups, _ = sender
    assert value.endpoints == ["http://one:8000", "http://two:8001"]
    assert not value.session.trust_env
    assert [g.info["world_size"] for g in groups] == [3, 3]
    assert [g.info["rank_offset"] for g in groups] == [1, 1]
    assert [g.info["master_port"] for g in groups] == [29001, 29002]
    init_bodies = [
        body for _, url, body, _ in calls if url.endswith("init_weight_transfer_engine")
    ]
    assert [body["init_info"] for body in init_bodies] == [g.info for g in groups]
    value.close()
    for group in groups:
        group.destroy.assert_called_once()


def test_update_metadata_packed_geometry_and_finish_before_resume(sender):
    value, calls, groups, sends = sender
    weights = [
        ("model.a", torch.ones(2, 3)),
        ("model.b", torch.ones(4, dtype=torch.bfloat16)),
    ]
    value.update(iter(weights), 12)
    assert len(sends) == 4
    updates = [
        (url, body) for _, url, body, _ in calls if url.endswith("/update_weights")
    ]
    assert len(updates) == 4
    for (_, body), (batch, args) in zip(updates, sends):
        info = body["update_info"]
        assert info["names"] == [batch[0][0]]
        assert info["shapes"] == [list(batch[0][1].shape)]
        assert info["dtype_names"] == [str(batch[0][1].dtype).removeprefix("torch.")]
        assert info["packed_buffer_size_bytes"] == args["packed_buffer_size_bytes"]
        assert info["packed_num_buffers"] == args["packed_num_buffers"] == 2
    routes = [url.rsplit("/", 1)[-1] for _, url, _, _ in calls]
    assert routes[-4:] == [
        "finish_weight_update",
        "finish_weight_update",
        "resume",
        "resume",
    ]
    pauses = [params for _, url, _, params in calls if url.endswith("/pause")]
    assert pauses == [{"mode": "wait", "clear_cache": "true"}] * 2
    assert not value.failed


def test_http_background_error_is_propagated_and_does_not_resume(sender, monkeypatch):
    value, calls, _, _ = sender
    request = value._request

    def failing_request(endpoint, route, *args, **kwargs):
        if route == "update_weights":
            raise RuntimeError("receiver failed")
        return request(endpoint, route, *args, **kwargs)

    monkeypatch.setattr(value, "_request", failing_request)
    with pytest.raises(RuntimeError, match="receiver failed"):
        value.update(iter([("a", torch.ones(2))]), 12)
    assert value.failed
    assert not any(
        url.endswith("/resume") or url.endswith("finish_weight_update")
        for _, url, _, _ in calls
    )
    with pytest.raises(RuntimeError, match="Previous"):
        value.update(iter([]), 13)


@pytest.mark.parametrize("rank", [0, 1])
def test_all_actor_ranks_exhaust_export_iterator_and_restore_offload(rank):
    exported = []

    def iterator():
        for i in range(3):
            exported.append(i)
            yield str(i), torch.ones(1)

    engine = SimpleNamespace(
        get_per_tensor_param=Mock(return_value=(iterator(), None)),
        is_param_offload_enabled=True,
        to=Mock(),
    )
    sender = SimpleNamespace(update=Mock(side_effect=lambda items, step: list(items)))
    worker = SimpleNamespace(
        rank=rank,
        actor=SimpleNamespace(engine=engine),
        _speco_external_vllm_sender=sender,
    )
    sync.update_worker_weights(worker, 9)
    assert exported == [0, 1, 2]
    assert sender.update.call_count == int(rank == 0)
    engine.to.assert_called_once_with("cpu", model=True, optimizer=False, grad=False)


def test_sender_error_still_drains_actor_export():
    exported = []

    def iterator():
        for i in range(3):
            exported.append(i)
            yield str(i), torch.ones(1)

    engine = SimpleNamespace(
        get_per_tensor_param=lambda **kw: (iterator(), None),
        is_param_offload_enabled=False,
    )
    worker = SimpleNamespace(
        rank=0,
        actor=SimpleNamespace(engine=engine),
        _speco_external_vllm_sender=SimpleNamespace(
            update=Mock(side_effect=RuntimeError("HTTP failed"))
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP failed"):
        sync.update_worker_weights(worker, 2)
    assert exported == [0, 1, 2]


def test_driver_disabled_and_enabled_lifecycle():
    group = Mock()
    disabled = sync.initialize_external_vllm_weight_sync({}, actor_worker_group=group)
    sync.update_external_vllm_weights(disabled, global_step=0)
    sync.close_external_vllm_weight_sync(disabled)
    assert not group.mock_calls
    handle = sync.initialize_external_vllm_weight_sync(
        {"enabled": True}, actor_worker_group=group
    )
    assert handle.last_synced_step is None
    sync.update_external_vllm_weights(handle, global_step=14)
    assert handle.last_synced_step == 14
    group.update_external_vllm_weights.assert_called_once_with(14)
    sync.close_external_vllm_weight_sync(handle)
    group.close_external_vllm_weight_sync.assert_called_once()
    assert not handle.enabled


def test_driver_does_not_record_failed_update():
    handle = sync.ExternalVllmWeightSync(True, Mock())
    handle.actor_worker_group.update_external_vllm_weights.side_effect = RuntimeError(
        "failed"
    )
    with pytest.raises(RuntimeError):
        sync.update_external_vllm_weights(handle, global_step=2)
    assert handle.last_synced_step is None


@pytest.mark.parametrize("rank", [0, 1])
def test_norm_export_comes_from_same_weight_iterator(rank):
    norm = torch.tensor([2.0, 3.0])
    engine = SimpleNamespace(
        get_per_tensor_param=lambda **kw: (
            iter([("model.norm.weight", norm), ("lm_head.weight", torch.ones(2, 2))]),
            None,
        ),
        is_param_offload_enabled=False,
    )
    sent = []

    def send(items, step):
        sent.extend(list(items))
        norm.fill_(99)  # Export buffers may subsequently be reused.

    worker = SimpleNamespace(
        rank=rank,
        actor=SimpleNamespace(engine=engine),
        _speco_external_vllm_sender=SimpleNamespace(update=send),
        _speco_final_norm_names=("model.norm.weight",),
    )
    result = sync.update_worker_weights(worker, 4)
    if rank == 0:
        torch.testing.assert_close(
            result["model.norm.weight"], torch.tensor([2.0, 3.0])
        )
        assert [name for name, _ in sent] == ["model.norm.weight", "lm_head.weight"]
    else:
        assert result == {} and not sent


@pytest.mark.parametrize("fail_norm", [False, True])
def test_driver_updates_producer_norm_before_marking_sync_complete(
    monkeypatch, fail_norm
):
    events = []
    state = {"model.norm.weight": torch.tensor([2.0, 3.0])}

    def update_norm(values):
        assert values is state
        events.append("norm")
        if fail_norm:
            raise RuntimeError("norm update failed")

    producer = SimpleNamespace(update_final_norm=SimpleNamespace(remote=update_norm))

    def update_weights(step):
        events.append("vllm")
        return [state, {}]

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))
    handle = sync.ExternalVllmWeightSync(
        True,
        SimpleNamespace(update_external_vllm_weights=update_weights),
        feature_producer=producer,
        final_norm_names=("model.norm.weight",),
    )
    if fail_norm:
        with pytest.raises(RuntimeError, match="norm update failed"):
            sync.update_external_vllm_weights(handle, global_step=9)
        assert handle.last_synced_step is None
    else:
        sync.update_external_vllm_weights(handle, global_step=9)
        assert handle.last_synced_step == 9
    assert events == ["vllm", "norm"]


def test_worker_rpc_returns_norm_payload(monkeypatch):
    import ast
    from pathlib import Path

    source = Path(__file__).parents[2] / "verl_speco/integration/rollout_publish.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_external_vllm_weights"
    )
    method.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    payload = {"model.norm.weight": torch.ones(2)}
    monkeypatch.setattr(sync, "update_worker_weights", lambda worker, step: payload)
    assert namespace["update_external_vllm_weights"](object(), 5) is payload


def test_worker_rejects_npu_before_nccl_init(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker = SimpleNamespace(
        config=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2")), rank=0
    )
    with pytest.raises(RuntimeError, match="NPU/HCCL"):
        sync.initialize_worker_weight_sync(worker, {})


def test_unpacked_update_omits_packed_geometry(sender):
    value, calls, _, sends = sender
    value.packed = False
    value.update(iter([("a", torch.ones(3))]), 1)
    for _, _, body, _ in calls:
        if body and "update_info" in body:
            assert body["update_info"]["packed"] is False
            assert "packed_buffer_size_bytes" not in body["update_info"]
    assert all("packed_buffer_size_bytes" not in args for _, args in sends)


def test_oversized_weight_expands_packed_buffer(sender):
    value, calls, _, _ = sender
    value.bucket_bytes = 4
    value.update(iter([("a", torch.ones(10))]), 1)
    for _, _, body, _ in calls:
        if body and "update_info" in body:
            assert body["update_info"]["packed_buffer_size_bytes"] == 40


def test_staging_batches_own_exported_views(monkeypatch):
    # Redirect just the CUDA placement to CPU; exercise actual staging logic.
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    value = sync.ExternalVllmWeightSender.__new__(sync.ExternalVllmWeightSender)
    value.bucket_bytes = 16

    def exporter():
        reused = torch.zeros(2)
        for i in range(3):
            reused.fill_(i)
            yield str(i), reused

    batches = list(value._batches(exporter()))
    assert [len(batch) for batch in batches] == [2, 1]
    assert [tensor.tolist() for batch in batches for _, tensor in batch] == [
        [0, 0],
        [1, 1],
        [2, 2],
    ]


def test_checkpoint_hook_syncs_external_before_rollout_and_restores_methods():
    # Exercise the actual hook without importing the optional verl stack.
    import ast
    from contextlib import contextmanager
    from pathlib import Path
    from types import MethodType

    source = Path(__file__).parents[2] / "verl_speco/trainer/speco_ray_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpecoRayPPOTrainer"
    )
    hook = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_speco_online_fit_hooks"
    )
    namespace = {
        "contextmanager": contextmanager,
        "MethodType": MethodType,
        "DataProto": object,
        "update_external_vllm_weights": sync.update_external_vllm_weights,
    }
    exec(
        compile(ast.Module(body=[hook], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    events = []
    handle = sync.ExternalVllmWeightSync(
        True,
        SimpleNamespace(
            update_external_vllm_weights=lambda step: events.append(("external", step))
        ),
    )
    def original_update(*args, **kwargs):
        events.append(("rollout", 7))
    manager = SimpleNamespace(update_weights=original_update)
    rollout = SimpleNamespace(generate_sequences=lambda: None)
    trainer = SimpleNamespace(
        global_steps=7,
        checkpoint_manager=manager,
        _speco_external_vllm_weight_sync=handle,
        _speco_rollout_generation_target=lambda: rollout,
        _compute_old_log_prob=lambda *a: None,
        _update_actor=lambda *a: None,
        _speco_oldlogprob_collection_requested=lambda: False,
        _speco_vllm_collection_requested=lambda: False,
        _speco_oldlogprob_entropy_hook_enabled=lambda: False,
        _speco_wait_pending_drafter_publish=lambda: None,
    )
    with namespace["_speco_online_fit_hooks"](trainer):
        manager.update_weights(7)
    assert events == [("external", 7), ("rollout", 7)]
    assert handle.last_synced_step == 7
    assert manager.update_weights is original_update
