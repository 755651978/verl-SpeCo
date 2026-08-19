# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from verl_speco.trainer.mooncake_transfer import (  # noqa: E402
    MooncakeTensorStore,
    MooncakeTransferConfig,
    _parse_size,
)


class _RawStore:
    objects = {}

    def setup(self, **kwargs):
        self.setup_kwargs = kwargs
        return 0

    def put(self, key, payload):
        self.objects[key] = payload
        return 0

    def get(self, key):
        return self.objects.get(key)

    def remove(self, key, force):
        self.objects.pop(key, None)

    def close(self):
        return None


def test_mooncake_tensor_store_roundtrip(monkeypatch):
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = _RawStore
    mooncake_module = types.ModuleType("mooncake")
    mooncake_module.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    config = MooncakeTransferConfig(
        local_hostname="localhost",
        metadata_server="P2PHANDSHAKE",
        master_server_address="127.0.0.1:50051",
        global_segment_size=_parse_size("64MB"),
        local_buffer_size=_parse_size("128MB"),
        protocol="tcp",
        device_name="",
        get_timeout=1,
        get_poll_interval=0.01,
    )
    store = MooncakeTensorStore(config)
    expected = {
        "token_ids": torch.arange(4),
        "hidden_states": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
    }

    metadata = store.put("sample", expected)
    actual = store.get("sample")

    assert metadata["tensor_shapes"]["hidden_states"] == (2, 3, 4)
    assert torch.equal(actual["token_ids"], expected["token_ids"])
    assert torch.equal(actual["hidden_states"], expected["hidden_states"])
    store.remove("sample")
    assert "sample" not in _RawStore.objects
