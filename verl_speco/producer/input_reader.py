# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming JSONL input and token preparation for the standalone Producer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


@dataclass(frozen=True)
class InputRecord:
    sequence_no: int
    sample_id: str
    prompt: str
    response: str
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class TokenizedRequest:
    sequence_no: int
    sample_id: str
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    feature_positions: torch.Tensor
    draft_position_ids: torch.Tensor
    source_metadata: dict[str, Any]

    @property
    def prompt_token_ids(self) -> list[int]:
        feature_end = int(self.feature_positions[-1].item()) + 1
        return self.input_ids[:feature_end].detach().cpu().long().tolist()


def iter_input_records(path: str | os.PathLike[str]) -> Iterator[InputRecord]:
    """Yield one strict prompt/response record per non-empty JSONL line."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Producer input JSONL not found: {input_path}")
    sequence_no = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON object at {input_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Producer input at {input_path}:{line_number} must be a JSON object"
                )
            prompt = payload.get("prompt")
            response = payload.get("response")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Producer input at {input_path}:{line_number} requires string field 'prompt'"
                )
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"Producer input at {input_path}:{line_number} requires non-empty string field 'response'"
                )
            sample_id = payload.get("sample_id", f"train-{sequence_no:06d}")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"Producer input at {input_path}:{line_number} has invalid sample_id"
                )
            source_metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"prompt", "response", "sample_id"}
            }
            yield InputRecord(
                sequence_no=sequence_no,
                sample_id=sample_id,
                prompt=prompt,
                response=response,
                source_metadata=source_metadata,
            )
            sequence_no += 1


def build_loss_mask(input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    sequence_length = int(input_ids.numel())
    if prompt_length < 0 or prompt_length > sequence_length:
        raise ValueError(
            f"prompt_length must be within [0, {sequence_length}], got {prompt_length}"
        )
    mask = torch.ones(sequence_length, dtype=torch.float32)
    mask[:prompt_length] = 0
    return mask


def tokenize_record(
    record: InputRecord,
    tokenizer: Any,
    config: Mapping[str, Any] | Any,
) -> TokenizedRequest:
    """Tokenize existing prompt/response text without generating new tokens."""

    prompt_ids = _token_ids(tokenizer(record.prompt, add_special_tokens=False))
    full_ids = _token_ids(
        tokenizer(record.prompt + record.response, add_special_tokens=False)
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Producer sample {record.sample_id!r} has an unstable tokenizer boundary "
            "between prompt and response; prompt token IDs are not a prefix of full token IDs"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"Producer sample {record.sample_id!r} produced no response tokens"
        )
    input_ids = torch.tensor(full_ids, dtype=torch.int64)
    if int(input_ids.numel()) <= 0:
        raise ValueError(
            f"Producer sample {record.sample_id!r} produced no input tokens"
        )

    max_sequence_length = int(_config_value(config, "max_sequence_length", 0) or 0)
    if max_sequence_length > 0 and int(input_ids.numel()) > max_sequence_length:
        raise ValueError(
            f"Producer sample {record.sample_id!r} has {int(input_ids.numel())} tokens, "
            f"exceeding max_sequence_length={max_sequence_length}"
        )
    loss_mask = build_loss_mask(input_ids, len(prompt_ids))
    position_ids = torch.arange(int(input_ids.numel()), dtype=torch.int64)

    feature_start = max(len(prompt_ids) - 1, 0)
    feature_end = int(input_ids.numel())
    max_feature_length = int(_config_value(config, "max_feature_length", 0) or 0)
    if max_feature_length == 1:
        raise ValueError("max_feature_length must be 0 or at least 2")
    if max_feature_length > 1:
        feature_end = min(feature_start + max_feature_length, feature_end)
    feature_positions = torch.arange(feature_start, feature_end, dtype=torch.int64)
    if int(feature_positions.numel()) <= 0:
        raise ValueError(
            f"Producer sample {record.sample_id!r} has an empty feature window"
        )
    draft_position_ids = position_ids[feature_start:feature_end] + 1
    return TokenizedRequest(
        sequence_no=record.sequence_no,
        sample_id=record.sample_id,
        input_ids=input_ids,
        loss_mask=loss_mask,
        position_ids=position_ids,
        feature_positions=feature_positions,
        draft_position_ids=draft_position_ids,
        source_metadata=dict(record.source_metadata),
    )


def _token_ids(encoding: Any) -> list[int]:
    value = (
        encoding.get("input_ids")
        if isinstance(encoding, Mapping)
        else encoding.input_ids
    )
    if value is None:
        raise ValueError("Tokenizer result is missing input_ids")
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer input_ids must be a tensor, list, or tuple")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Tokenizer returned more than one sequence for one input")
        value = value[0]
    return [int(token_id) for token_id in value]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


__all__ = [
    "InputRecord",
    "TokenizedRequest",
    "build_loss_mask",
    "iter_input_records",
    "tokenize_record",
]
