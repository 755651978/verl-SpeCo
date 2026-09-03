# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from verl_speco.producer.input_reader import build_rollout_prefill_request
from verl_speco.producer.vllm_feature_client import RawVllmFeature
from verl_speco.producer.vllm_feature_producer import (
    VllmFeatureProducerCore,
    build_feature_contract,
)


class _FakeClientPool:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def prefill(self, request) -> RawVllmFeature:
        assert self.started
        token_ids = torch.tensor(request.prompt_token_ids)
        hidden = torch.arange(token_ids.numel() * 3 * 4, dtype=torch.float32).reshape(
            token_ids.numel(), 3, 4
        )
        return RawVllmFeature(
            payload={"token_ids": token_ids, "hidden_states": hidden},
            temporary_path="",
            endpoint_url="http://vllm.test/v1",
            byte_size=int(hidden.numel() * hidden.element_size()),
        )

    async def close(self) -> None:
        self.closed = True


def test_core_converts_rollout_prefill_to_draft_feature() -> None:
    asyncio.run(_test_core_converts_rollout_prefill_to_draft_feature())


async def _test_core_converts_rollout_prefill_to_draft_feature() -> None:
    request = build_rollout_prefill_request(
        sequence_no=0,
        sample_id="rollout-0",
        prompt_token_ids=[10, 11],
        response_token_ids=[20, 21, 22, 23, 24],
        feature_positions=[2, 3, 4],
        config={"max_sequence_length": 16},
    )
    pool = _FakeClientPool()
    core = VllmFeatureProducerCore(
        pool,
        build_feature_contract(
            {
                "target_layer_ids": [1, 9],
                "hidden_dtype": "float32",
                "target_model_id": "target",
            },
            {
                "speculative_algorithm": "EAGLE3",
                "training": {"use_logits": False},
            },
            source="test",
        ),
        final_norm=torch.nn.RMSNorm(4, eps=1e-6).requires_grad_(False),
    )

    await core.start()
    produced = await core.produce_one(request)
    await core.close()

    assert produced.request is request
    assert produced.sample.input_ids.tolist() == [20, 21, 22]
    assert produced.sample.loss_mask.tolist() == [1.0, 1.0, 1.0]
    assert produced.sample.hidden_states.shape == (3, 12)
    assert produced.sample.position_ids.tolist() == [3, 4, 5]
    assert produced.sample.metadata["source"] == "test"
    raw = torch.arange(6 * 3 * 4, dtype=torch.float32).reshape(6, 3, 4)[2:5]
    torch.testing.assert_close(
        produced.sample.hidden_states[:, :8], raw[:, :2].flatten(1)
    )
    torch.testing.assert_close(
        produced.sample.hidden_states[:, 8:], core.final_norm(raw[:, 2])
    )
    assert pool.closed


def test_ray_producer_refreshes_norm_for_next_batch(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(put=lambda tensor: tensor))
    path = Path(__file__).parents[2] / "verl_speco/producer/ray_feature_producer.py"
    spec = importlib.util.spec_from_file_location("_norm_producer_actor_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    norm = torch.nn.RMSNorm(4, eps=1e-6).requires_grad_(False)
    norm._speco_checkpoint_name = "model.norm"
    monkeypatch.setattr(module, "load_vllm_final_norm", lambda *a, **kw: norm)
    monkeypatch.setattr(module, "build_client_pool", lambda cfg: _FakeClientPool())
    monkeypatch.setattr(module, "delete_temporary_result", lambda raw: None)
    actor = module.VllmFeatureProducerActor(
        {
            "target_layer_ids": [1, 9],
            "hidden_dtype": "float32",
            "target_model_id": "target",
            "use_object_store": False,
        },
        {"speculative_algorithm": "EAGLE3", "training": {"use_logits": False}},
    )
    assert actor.get_final_norm_names() == ["model.norm.weight"]
    request = build_rollout_prefill_request(
        sequence_no=0,
        sample_id="sample",
        prompt_token_ids=[10, 11],
        response_token_ids=[20, 21, 22, 23, 24],
        feature_positions=[2, 3, 4],
        config={"max_sequence_length": 16},
    )

    async def run():
        before = (await actor.produce_batch([request]))[0].sample.hidden_states.clone()
        actor.update_final_norm({"model.norm.weight": torch.full((4,), 2.0)})
        after = (await actor.produce_batch([request]))[0].sample.hidden_states
        torch.testing.assert_close(before[:, :8], after[:, :8])
        torch.testing.assert_close(before[:, 8:] * 2, after[:, 8:])
        await actor.close()

    asyncio.run(run())
    with pytest.raises(ValueError, match="expected final norm"):
        actor.update_final_norm({})
