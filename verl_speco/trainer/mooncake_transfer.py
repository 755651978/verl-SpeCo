# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Small optional Mooncake client used by standalone target-feature replay.

The payload is stored as one safetensors object.  A single object makes the
producer response atomic and avoids the file creation/locking protocol used by
``vllm_file``.  Mooncake is imported lazily so normal and online training do
not acquire a runtime dependency on it.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _parse_size(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().upper()
    multipliers = {
        "TB": 1024**4,
        "GB": 1024**3,
        "MB": 1024**2,
        "KB": 1024,
        "T": 1024**4,
        "G": 1024**3,
        "M": 1024**2,
        "K": 1024,
        "B": 1,
    }
    for suffix in sorted(multipliers, key=len, reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multipliers[suffix])
    return int(text)


@dataclass(frozen=True)
class MooncakeTransferConfig:
    local_hostname: str
    metadata_server: str
    master_server_address: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    get_timeout: float
    get_poll_interval: float

    @classmethod
    def from_mapping(cls, config: Any | None = None) -> "MooncakeTransferConfig":
        config = config or {}

        def value(name: str, default: Any) -> Any:
            getter = getattr(config, "get", None)
            if callable(getter):
                return getter(name, default)
            return getattr(config, name, default)

        master = str(
            value(
                "master_server_address",
                os.getenv("MOONCAKE_MASTER_SERVER", "127.0.0.1:50051"),
            )
        )
        master_host = master.rsplit(":", 1)[0]
        return cls(
            local_hostname=str(
                value(
                    "local_hostname",
                    os.getenv("MOONCAKE_LOCAL_HOSTNAME", socket.gethostname()),
                )
            ),
            metadata_server=str(
                value(
                    "metadata_server",
                    os.getenv(
                        "MOONCAKE_METADATA_SERVER",
                        f"http://{master_host}:8090/metadata",
                    ),
                )
            ),
            master_server_address=master,
            global_segment_size=_parse_size(
                value(
                    "global_segment_size",
                    os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "4GB"),
                )
            ),
            local_buffer_size=_parse_size(
                value(
                    "local_buffer_size",
                    os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", "1GB"),
                )
            ),
            protocol=str(value("protocol", os.getenv("MOONCAKE_PROTOCOL", "tcp"))),
            device_name=str(
                value("device_name", os.getenv("MOONCAKE_DEVICE_NAME", ""))
            ),
            get_timeout=float(value("get_timeout", 120.0)),
            get_poll_interval=max(float(value("get_poll_interval", 0.02)), 0.001),
        )

    def export_environment(self) -> None:
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = self.local_hostname
        os.environ["MOONCAKE_METADATA_SERVER"] = self.metadata_server
        os.environ["MOONCAKE_MASTER_SERVER"] = self.master_server_address
        os.environ["MOONCAKE_GLOBAL_SEGMENT_SIZE"] = str(self.global_segment_size)
        os.environ["MOONCAKE_LOCAL_BUFFER_SIZE"] = str(self.local_buffer_size)
        os.environ["MOONCAKE_PROTOCOL"] = self.protocol
        os.environ["MOONCAKE_DEVICE_NAME"] = self.device_name
        if self.protocol.lower() == "tcp":
            os.environ.setdefault("MC_STORE_MEMCPY", "0")


class MooncakeTensorStore:
    """Store and retrieve a tensor dictionary as one Mooncake object."""

    def __init__(self, config: MooncakeTransferConfig):
        self.config = config
        self._store: Any | None = None

    def setup(self) -> None:
        if self._store is not None:
            return
        self.config.export_environment()
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise RuntimeError(
                "Mooncake replay requires mooncake-transfer-engine "
                "(use mooncake-transfer-engine-npu on Ascend)"
            ) from exc
        store = MooncakeDistributedStore()
        result = store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result not in (None, 0):
            raise RuntimeError(f"Mooncake client setup failed with code {result}")
        self._store = store

    def put(self, key: str, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.setup()
        assert self._store is not None
        from safetensors.torch import save

        cpu_tensors = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in tensors.items()
        }
        payload = save(cpu_tensors)
        result = self._store.put(key, payload)
        if result not in (None, 0):
            raise RuntimeError(f"Mooncake put failed for {key!r}: code={result}")
        return {
            "mooncake_key": key,
            "tensor_shapes": {
                name: tuple(tensor.shape) for name, tensor in cpu_tensors.items()
            },
            "tensor_dtypes": {
                name: str(tensor.dtype).removeprefix("torch.")
                for name, tensor in cpu_tensors.items()
            },
            "payload_bytes": len(payload),
        }

    def get(self, key: str) -> dict[str, torch.Tensor]:
        self.setup()
        assert self._store is not None
        from safetensors.torch import load

        deadline = time.monotonic() + self.config.get_timeout
        while True:
            payload = self._store.get(key)
            if payload is not None:
                return dict(load(bytes(payload)))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Mooncake object {key!r} was unavailable for "
                    f"{self.config.get_timeout:.1f}s"
                )
            time.sleep(self.config.get_poll_interval)

    def remove(self, key: str) -> None:
        if self._store is None:
            return
        try:
            remove = getattr(self._store, "remove", None)
            if callable(remove):
                remove(key, True)
            else:
                self._store.batch_remove([key], force=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to remove Mooncake object %s", key, exc_info=True)

    def close(self) -> None:
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._store = None
