# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Thin Ray actor wrapper around the reusable vLLM feature Producer core."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import ray
import torch

from verl_speco.producer.input_reader import TokenizedRequest
from verl_speco.producer.vllm_feature_client import delete_temporary_result
from verl_speco.producer.vllm_feature_producer import (
    VllmFeatureProducerCore,
    build_client_pool,
    build_feature_contract,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.target_feature_replay import load_vllm_final_norm


@dataclass(frozen=True)
class ProducedFeatureResult:
    request_id: str
    request: TokenizedRequest
    sample: DraftFeatureSample | None
    hidden_states_ref: Any | None = None
    error: str | None = None


class VllmFeatureProducerActor:
    """Own one client pool while leaving scheduling and routing to the driver."""

    def __init__(
        self,
        producer_config: Mapping[str, Any],
        drafter_config: Mapping[str, Any],
    ) -> None:
        producer_config = dict(producer_config)
        drafter_config = dict(drafter_config)
        self._use_object_store = bool(producer_config.get("use_object_store", True))
        contract = build_feature_contract(
            producer_config, drafter_config, source="cotrain_vllm_producer"
        )
        norm = None
        self._norm_names = {}
        if contract.hidden_states_layout.endswith("_plus_last"):
            norm = load_vllm_final_norm(
                contract.target_model_id,
                dtype=contract.dtype,
                trust_remote_code=bool(producer_config.get("trust_remote_code", False)),
            )
            self._norm_names = {
                f"{norm._speco_checkpoint_name}.{key}": key for key in norm.state_dict()
            }
        self.core = VllmFeatureProducerCore(
            build_client_pool(producer_config), contract, final_norm=norm
        )
        self._started = False

    def get_final_norm_names(self) -> list[str]:
        return list(self._norm_names)

    def update_final_norm(self, state: Mapping[str, torch.Tensor]) -> None:
        """Apply the same norm weights that were just sent to the vLLM endpoint."""
        if set(state) != set(self._norm_names):
            raise ValueError(
                "Actor export does not contain the expected final norm tensors"
            )
        if self.core.final_norm is not None:
            self.core.final_norm.load_state_dict(
                {self._norm_names[name]: value for name, value in state.items()},
                strict=True,
            )

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.core.start()
            self._started = True

    async def produce_batch(
        self, requests: Sequence[TokenizedRequest]
    ) -> list[ProducedFeatureResult]:
        await self._ensure_started()
        produced = await self.core.produce_batch(requests)
        results: list[ProducedFeatureResult] = []
        for request, item in zip(requests, produced, strict=True):
            if isinstance(item, BaseException):
                results.append(
                    ProducedFeatureResult(
                        request_id=request.sample_id,
                        request=request,
                        sample=None,
                        error=f"{type(item).__name__}: {item}",
                    )
                )
                continue
            try:
                if self._use_object_store:
                    hidden_ref = ray.put(item.sample.hidden_states)
                    sample = replace(
                        item.sample,
                        hidden_states=torch.empty(0, dtype=torch.uint8),
                    )
                else:
                    hidden_ref = None
                    sample = item.sample
                results.append(
                    ProducedFeatureResult(
                        request_id=request.sample_id,
                        request=request,
                        sample=sample,
                        hidden_states_ref=hidden_ref,
                    )
                )
            finally:
                await asyncio.to_thread(delete_temporary_result, item.raw)
        return results

    async def close(self) -> None:
        if self._started:
            await self.core.close()
            self._started = False


__all__ = ["ProducedFeatureResult", "VllmFeatureProducerActor"]
