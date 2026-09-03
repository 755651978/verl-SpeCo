# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Reusable vLLM hidden-state production independent of TQ and Ray."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_drafter_hidden_states_layout,
)
from verl_speco.producer.input_reader import TokenizedRequest
from verl_speco.producer.vllm_feature_client import (
    RawVllmFeature,
    VllmEndpoint,
    VllmFeatureClientPool,
    delete_temporary_result,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.target_feature_replay import (
    FeatureContract,
    feature_from_vllm_payload,
)


@dataclass(frozen=True)
class ProducedFeature:
    request: TokenizedRequest
    raw: RawVllmFeature
    sample: DraftFeatureSample


def parse_hidden_dtype(value: Any) -> torch.dtype:
    name = str(value or "bfloat16").strip().lower().removeprefix("torch.")
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype = getattr(torch, aliases.get(name, name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported vLLM feature dtype={value!r}")
    return dtype


def build_feature_contract(
    producer_config: Mapping[str, Any],
    drafter_config: Mapping[str, Any],
    *,
    source: str,
) -> FeatureContract:
    algorithm = str(drafter_config.get("speculative_algorithm", "")).strip().upper()
    layer_ids = producer_config.get("target_layer_ids")
    if not algorithm:
        raise ValueError("drafter.speculative_algorithm must not be empty")
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError("vLLM feature target_layer_ids must be a non-empty list")
    training_config = drafter_config.get("training") or {}
    return FeatureContract(
        algorithm=algorithm,
        target_layer_ids=[int(value) for value in layer_ids],
        hidden_states_layout=resolve_drafter_hidden_states_layout(
            algorithm, drafter_config
        ),
        dtype=parse_hidden_dtype(producer_config.get("hidden_dtype", "bfloat16")),
        target_model_id=str(producer_config.get("target_model_id") or ""),
        target_model_revision=producer_config.get("target_model_revision"),
        tokenizer_fingerprint=str(producer_config.get("tokenizer_fingerprint") or ""),
        use_logits=bool(training_config.get("use_logits", False)),
        source=source,
        require_full_alignment=True,
    )


def build_client_pool(config: Mapping[str, Any]) -> VllmFeatureClientPool:
    endpoints = config.get("endpoints", config.get("vllm_endpoints"))
    if not isinstance(endpoints, (list, tuple)) or not endpoints:
        raise ValueError("vLLM feature endpoints must be a non-empty list")
    concurrency = int(config.get("per_endpoint_concurrency", 1) or 1)
    return VllmFeatureClientPool(
        [VllmEndpoint(str(url).rstrip("/"), concurrency) for url in endpoints],
        model=str(config.get("model", config.get("vllm_model")) or ""),
        max_inflight_requests=int(config.get("max_inflight_requests", 1) or 1),
        request_timeout=float(
            config.get("request_timeout_seconds", config.get("request_timeout", 600))
            or 600
        ),
    )


class VllmFeatureProducerCore:
    """Convert tokenized requests into normalized draft features."""

    def __init__(
        self,
        client_pool: VllmFeatureClientPool,
        feature_contract: FeatureContract,
        *,
        final_norm: torch.nn.Module | None = None,
    ) -> None:
        self.client_pool = client_pool
        self.feature_contract = feature_contract
        self.final_norm = final_norm

    async def start(self) -> None:
        await self.client_pool.start()

    async def produce_one(self, request: TokenizedRequest) -> ProducedFeature:
        raw = await self.client_pool.prefill(request)
        try:
            sample = feature_from_vllm_payload(
                raw, request, self.feature_contract, final_norm=self.final_norm
            )
        except BaseException:
            await asyncio.to_thread(delete_temporary_result, raw)
            raise
        return ProducedFeature(request=request, raw=raw, sample=sample)

    async def produce_batch(
        self, requests: Sequence[TokenizedRequest]
    ) -> list[ProducedFeature | BaseException]:
        return list(
            await asyncio.gather(
                *(self.produce_one(request) for request in requests),
                return_exceptions=True,
            )
        )

    async def close(self) -> None:
        await self.client_pool.close()


__all__ = [
    "ProducedFeature",
    "VllmFeatureProducerCore",
    "build_client_pool",
    "build_feature_contract",
    "parse_hidden_dtype",
]
