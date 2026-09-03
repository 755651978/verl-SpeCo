# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""External vLLM 0.23 weight updates: actor-local NCCL, HTTP metadata only."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)


def _server_url(endpoint: str) -> str:
    parsed = urlsplit(str(endpoint).rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid external vLLM endpoint: {endpoint!r}")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.removesuffix("/v1"), "", "")
    )


@dataclass
class ExternalVllmWeightSync:
    enabled: bool = False
    actor_worker_group: Any = None
    last_synced_step: int | None = None
    feature_producer: Any = None
    final_norm_names: tuple[str, ...] = ()


def initialize_external_vllm_weight_sync(
    config: Mapping[str, Any] | None,
    *,
    actor_worker_group: Any,
    feature_producer: Any = None,
) -> ExternalVllmWeightSync:
    config = dict(config or {})
    if not config.get("enabled", False):
        return ExternalVllmWeightSync()
    actor_worker_group.init_external_vllm_weight_sync(config)
    return ExternalVllmWeightSync(
        True,
        actor_worker_group,
        feature_producer=feature_producer,
        final_norm_names=tuple(config.get("final_norm_names", ())),
    )


def update_external_vllm_weights(
    handle: ExternalVllmWeightSync | None,
    *,
    global_step: int,
) -> None:
    if handle is not None and handle.enabled:
        results = handle.actor_worker_group.update_external_vllm_weights(global_step)
        if handle.final_norm_names:
            import ray

            states = [result for result in results if result]
            if len(states) != 1 or set(states[0]) != set(handle.final_norm_names):
                raise RuntimeError(
                    "Missing rank-0 final norm export after vLLM weight update"
                )
            ray.get(handle.feature_producer.update_final_norm.remote(states[0]))
        handle.last_synced_step = int(global_step)


def close_external_vllm_weight_sync(
    handle: ExternalVllmWeightSync | None,
) -> None:
    if handle is not None and handle.enabled:
        try:
            handle.actor_worker_group.close_external_vllm_weight_sync()
        finally:
            handle.enabled = False


class ExternalVllmWeightSender:
    """Lives on actor rank 0, on its existing CUDA device, not on the driver."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import requests
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )
        from vllm.utils.network_utils import get_ip, get_open_port

        if not torch.cuda.is_available():
            raise RuntimeError(
                "External vLLM NCCL weight sync requires CUDA, not NPU/HCCL"
            )
        endpoints = config.get("endpoints") or []
        if isinstance(endpoints, str) or not endpoints:
            raise ValueError("External vLLM weight sync requires an endpoint list")
        self.endpoints = list(dict.fromkeys(_server_url(url) for url in endpoints))
        self.timeout = float(config.get("timeout_seconds", 600))
        self.bucket_bytes = int(config.get("bucket_size_mb", 256)) * 1024**2
        self.packed = bool(config.get("packed", True))
        self.num_buffers = int(config.get("packed_num_buffers", 2))
        if self.timeout <= 0 or self.bucket_bytes <= 0 or self.num_buffers <= 0:
            raise ValueError(
                "weight sync timeout, bucket size and buffer count must be positive"
            )
        self.session = requests.Session()
        self.session.trust_env = False
        self.engine = NCCLWeightTransferEngine
        self.groups: list[Any] = []
        self.failed = False
        address = str(config.get("master_address") or get_ip())
        try:
            for endpoint in self.endpoints:
                size = int(
                    self._request(endpoint, "get_world_size", method="GET")[
                        "world_size"
                    ]
                )
                if size < 1:
                    raise ValueError(f"Invalid vLLM world_size={size} at {endpoint}")
                info = {
                    "master_address": address,
                    "master_port": get_open_port(),
                    "rank_offset": 1,
                    "world_size": size + 1,
                }
                # Server init blocks waiting for the sender's NCCL rendezvous.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(
                        self._request,
                        endpoint,
                        "init_weight_transfer_engine",
                        {"init_info": info},
                    )
                    group = self.engine.trainer_init(info)
                    self.groups.append(group)
                    pending.result()
                if not group.available or group.disabled:
                    raise RuntimeError(
                        "vLLM PyNcclCommunicator is unavailable/disabled"
                    )
                logger.warning(
                    "[external vLLM weights] connected endpoint=%s workers=%s",
                    endpoint,
                    size,
                )
        except BaseException:
            self.close()
            raise

    def _request(self, endpoint, route, payload=None, *, method="POST", params=None):
        response = self.session.request(
            method,
            f"{endpoint}/{route}",
            json=payload,
            params=params,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"External vLLM {route} failed: HTTP {response.status_code} "
                f"endpoint={endpoint} body={response.text[:2000]}"
            )
        return response.json()

    def _batches(self, iterator):
        import torch

        batch, size = [], 0
        for name, tensor in iterator:
            if not torch.is_tensor(tensor):
                raise TypeError(f"Actor weight {name!r} is not a tensor")
            nbytes = tensor.numel() * tensor.element_size()
            if batch and size + nbytes > self.bucket_bytes:
                yield batch
                batch, size = [], 0
            # Exporters may yield views into reused MoE buffers. Own the small
            # staging batch until every endpoint finishes consuming it.
            batch.append(
                (
                    str(name),
                    tensor.detach()
                    .to(
                        device=torch.device("cuda", torch.cuda.current_device()),
                        copy=True,
                    )
                    .contiguous(),
                )
            )
            size += nbytes
            if size >= self.bucket_bytes:
                yield batch
                batch, size = [], 0
        if batch:
            yield batch

    def update(self, iterator, global_step: int) -> None:
        import torch

        if self.failed:
            raise RuntimeError(
                "Previous external vLLM weight update failed; restart services/job"
            )
        count = 0
        try:
            for endpoint in self.endpoints:
                self._request(
                    endpoint, "pause", params={"mode": "wait", "clear_cache": "true"}
                )
                self._request(
                    endpoint, "start_weight_update", {"is_checkpoint_format": True}
                )
            for batch in self._batches(iterator):
                info = {
                    "names": [name for name, _ in batch],
                    "dtype_names": [
                        str(t.dtype).removeprefix("torch.") for _, t in batch
                    ],
                    "shapes": [list(t.shape) for _, t in batch],
                    "packed": self.packed,
                }
                args = {"packed": self.packed}
                if self.packed:
                    # A single parameter cannot be split by vLLM's packed API.
                    info["packed_buffer_size_bytes"] = max(
                        self.bucket_bytes,
                        max(t.numel() * t.element_size() for _, t in batch),
                    )
                    info["packed_num_buffers"] = self.num_buffers
                    args.update(
                        {
                            k: info[k]
                            for k in ("packed_buffer_size_bytes", "packed_num_buffers")
                        }
                    )
                torch.cuda.synchronize()
                for endpoint, group in zip(self.endpoints, self.groups, strict=True):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending = pool.submit(
                            self._request,
                            endpoint,
                            "update_weights",
                            {"update_info": info},
                        )
                        self.engine.trainer_send_weights(
                            iter(batch), {"group": group, **args}
                        )
                        torch.cuda.synchronize()
                        # Unlike Thread.join(), result() propagates HTTP errors.
                        pending.result()
                count += len(batch)
            if not count:
                raise RuntimeError("Actor exported no weights")
            # Do not resume any endpoint until every endpoint has finished.
            for endpoint in self.endpoints:
                self._request(endpoint, "finish_weight_update", {})
            for endpoint in self.endpoints:
                self._request(endpoint, "resume")
            logger.warning(
                "[external vLLM weights] updated step=%s tensors=%s endpoints=%s",
                global_step,
                count,
                len(self.endpoints),
            )
        except BaseException:
            self.failed = True
            logger.exception(
                "External vLLM update failed; endpoints are not automatically resumed"
            )
            raise

    def close(self) -> None:
        try:
            for group in self.groups:
                group.destroy()
        finally:
            self.groups.clear()
            self.session.close()


def initialize_worker_weight_sync(worker: Any, config: Mapping[str, Any]) -> None:
    import torch

    strategy = str(worker.config.actor.strategy).lower()
    if strategy not in {"fsdp", "fsdp2", "veomni"}:
        raise ValueError(
            "External vLLM sync requires a full HF weight iterator (FSDP/FSDP2/VeOmni)"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("External vLLM NCCL sync supports CUDA only, not NPU/HCCL")
    if getattr(worker, "_speco_external_vllm_sender", None) is not None:
        raise RuntimeError("External vLLM weight sync is already initialized")
    if worker.rank == 0:
        worker._speco_external_vllm_sender = ExternalVllmWeightSender(config)
    worker._speco_final_norm_names = tuple(config.get("final_norm_names", ()))


def update_worker_weights(worker: Any, global_step: int) -> dict[str, Any]:
    """Every actor rank consumes the export iterator (it contains collectives)."""
    engine = worker.actor.engine
    iterator, peft_config = engine.get_per_tensor_param(
        layered_summon=getattr(worker, "layered_summon", False),
        base_sync_done=True,
    )
    iterator = iter(iterator)
    norm_state = {}
    try:
        if peft_config is not None:
            raise ValueError(
                "External vLLM sync requires full/merged weights, not LoRA adapters"
            )
        if worker.rank == 0:
            sender = getattr(worker, "_speco_external_vllm_sender", None)
            if sender is None:
                raise RuntimeError("External vLLM weight sync is not initialized")

            def exported_weights():
                for name, tensor in iterator:
                    if name in getattr(worker, "_speco_final_norm_names", ()):
                        norm_state[name] = tensor.detach().cpu().clone()
                    yield name, tensor

            try:
                sender.update(exported_weights(), global_step)
                if set(norm_state) != set(
                    getattr(worker, "_speco_final_norm_names", ())
                ):
                    raise RuntimeError(
                        "Actor weight iterator missed target final norm parameters"
                    )
            except Exception:
                # Let peers finish export collectives before surfacing an HTTP
                # failure through the ONE_TO_ALL RPC. Never publish success.
                for _ in iterator:
                    pass
                raise
        else:
            for _ in iterator:
                pass
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        if engine.is_param_offload_enabled:
            engine.to("cpu", model=True, optimizer=False, grad=False)
    return norm_state


def close_worker_weight_sync(worker: Any) -> None:
    sender = getattr(worker, "_speco_external_vllm_sender", None)
    worker._speco_external_vllm_sender = None
    if sender is not None:
        sender.close()
