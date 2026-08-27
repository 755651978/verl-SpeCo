# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""vLLM connector that publishes extracted target features through Mooncake.

Load this module with ``kv_connector_module_path``.  It is intentionally kept
outside normal SpeCo imports because its API is tied to vLLM V1 internals.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

from verl_speco.trainer.mooncake_transfer import (
    MooncakeTensorStore,
    MooncakeTransferConfig,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


def _validate_vllm_version() -> None:
    try:
        from importlib.metadata import version

        from packaging.version import Version

        installed = Version(version("vllm"))
    except Exception:  # noqa: BLE001
        return
    if installed < Version("0.23.0"):
        raise RuntimeError(
            "SpeCoMooncakeHiddenStatesConnector requires vLLM >= 0.23.0; "
            f"found {installed}. The connector uses the V1 HMA hidden-state API."
        )


def _safe_key(key: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    return f"k{value}" if value and value[0].isdigit() else value


def _slot_mapping(
    block_ids: list[int], page_size: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    blocks = torch.tensor(block_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (blocks.unsqueeze(1) * page_size + offsets).flatten()[:num_tokens]


@dataclass
class _RequestMetadata:
    request_id: str
    token_ids: torch.Tensor
    block_ids: list[int] = field(default_factory=list)


@dataclass
class SpeCoMooncakeConnectorMetadata(KVConnectorMetadata):
    requests: list[_RequestMetadata] = field(default_factory=list)

    def add(self, request_id: str, token_ids: list[int], block_ids: list[int]) -> None:
        self.requests.append(
            _RequestMetadata(
                request_id=request_id,
                token_ids=torch.tensor(token_ids, dtype=torch.long),
                block_ids=list(block_ids),
            )
        )


class SpeCoMooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """Store-only connector for vLLM ``extract_hidden_states`` output."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        _validate_vllm_version()
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        speculative = vllm_config.speculative_config
        if speculative is None:
            raise ValueError(
                "SpeCoMooncakeHiddenStatesConnector requires extract_hidden_states"
            )
        hf_config = speculative.draft_model_config.hf_config
        self._layer_ids = list(
            getattr(hf_config, "eagle_aux_hidden_state_layer_ids", [])
        )
        self._hidden_size = int(vllm_config.model_config.get_hidden_size())
        self._training_layers = max(len(self._layer_ids) - 1, 1)
        self._cache_layers: list[str] = []
        self._cache_group_id = self._find_cache_group(kv_cache_config)
        self._active_requests: dict[str, Any] = {}
        self._request_blocks: dict[str, list[int]] = {}
        self._response_metadata: dict[str, dict[str, Any]] = {}
        configured_prefix = os.getenv("SPECO_MOONCAKE_KEY_PREFIX")
        self._key_prefix = _safe_key(
            configured_prefix or f"{socket.gethostname()}_{os.getpid()}"
        )
        self._store: MooncakeTensorStore | None = None
        self._store_setup_attempted = False
        self._tp_rank: int | None = None

    @staticmethod
    def _find_cache_group(kv_cache_config: "KVCacheConfig | None") -> int | None:
        if kv_cache_config is None:
            return None
        for index, group in enumerate(kv_cache_config.kv_cache_groups):
            if any("cache_only_layers" in name for name in group.layer_names):
                return index
        return None

    def _get_tp_rank(self) -> int:
        if self._tp_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self._tp_rank = int(get_tensor_model_parallel_rank())
            except Exception:  # noqa: BLE001
                self._tp_rank = 0
        return self._tp_rank

    def _ensure_store(self) -> MooncakeTensorStore | None:
        if self._store_setup_attempted:
            return self._store
        self._store_setup_attempted = True
        if self._get_tp_rank() != 0:
            return None
        try:
            store = MooncakeTensorStore(MooncakeTransferConfig.from_mapping())
            store.setup()
            self._store = store
        except Exception:  # noqa: BLE001
            logger.exception("Failed to initialize SpeCo Mooncake connector")
        return self._store

    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches)
        )
        self._cache_layers = list(layers)
        if len(self._cache_layers) != 1:
            raise RuntimeError(
                "Expected one extract_hidden_states cache layer, got "
                f"{self._cache_layers}"
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self._cache_layers:
            return
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )

        if not isinstance(attn_metadata, CacheOnlyAttentionMetadata):
            raise TypeError(
                "Expected CacheOnlyAttentionMetadata for extracted hidden states"
            )
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SpeCoMooncakeConnectorMetadata):
            raise TypeError("Unexpected connector metadata type")
        store = self._ensure_store()
        if store is None:
            return
        page_size = int(kv_layer.shape[1])
        for request in metadata.requests:
            num_tokens = int(request.token_ids.numel())
            positions = _slot_mapping(
                request.block_ids, page_size, num_tokens, kv_layer.device
            )
            if int(positions.numel()) < num_tokens:
                continue
            all_hidden = kv_layer.flatten(0, 1)[positions][:num_tokens].reshape(
                num_tokens, -1
            )
            split_at = self._training_layers * self._hidden_size
            training_hidden = all_hidden[:, :split_at].reshape(
                num_tokens, self._training_layers, self._hidden_size
            )
            last_hidden = all_hidden[:, -self._hidden_size :].unsqueeze(1)
            hidden_states = torch.cat((training_hidden, last_hidden), dim=1).to(
                torch.bfloat16
            )
            key = f"{self._key_prefix}_{_safe_key(request.request_id)}"
            result = store.put(
                key,
                {
                    "hidden_states": hidden_states,
                    "token_ids": request.token_ids,
                },
            )
            response = self._response_metadata.get(request.request_id)
            if response is not None:
                response.update(result)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens != 0:
            raise ValueError("SpeCo Mooncake connector is store-only")

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = SpeCoMooncakeConnectorMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            token_ids = request.prompt_token_ids or []
            group_id = self._cache_group_id
            if group_id is None:
                group_id = max(
                    range(len(request.block_ids)),
                    key=lambda index: len(request.block_ids[index]),
                )
                self._cache_group_id = group_id
            blocks = list(request.block_ids[group_id])
            metadata.add(request.req_id, token_ids, blocks)
            self._active_requests[request.req_id] = request
            self._request_blocks[request.req_id] = blocks
            self._response_metadata[request.req_id] = {
                "mooncake_key": (f"{self._key_prefix}_{_safe_key(request.req_id)}"),
                "input_ids_list": token_ids,
                "tensor_shapes": {
                    "hidden_states": (
                        len(token_ids),
                        self._training_layers + 1,
                        self._hidden_size,
                    ),
                    "token_ids": (len(token_ids),),
                },
                "tensor_dtypes": {
                    "hidden_states": "bfloat16",
                    "token_ids": "int64",
                },
            }

        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            if request_id not in self._active_requests:
                continue
            new_blocks = cached.new_block_ids[index]
            if new_blocks is not None:
                self._request_blocks[request_id].extend(
                    new_blocks[self._cache_group_id]
                )
            request = self._active_requests[request_id]
            metadata.add(
                request_id,
                request.prompt_token_ids or [],
                self._request_blocks[request_id],
            )
        return metadata

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = request.request_id
        self._active_requests.pop(request_id, None)
        self._request_blocks.pop(request_id, None)
        return False, self._response_metadata.pop(request_id, None)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids[0] if block_ids else [])

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "NHD"
