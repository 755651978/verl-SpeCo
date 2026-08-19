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
"""Wire protocol for standalone drafter samples stored in TransferQueue.

One TQ key represents one training sample.  Tensor payloads live in TQ fields;
small discovery attributes live in the TQ tag; richer metadata is JSON encoded
as a uint8 tensor so Producer and Consumer use one versioned contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from verl_speco.trainer.feature_store import DraftFeatureSample


PROTOCOL_SCHEMA_VERSION = 1
DRAFTER_TQ_PARTITION = "speco_drafter_features"
_REQUIRED_FIELDS = (
    "input_ids",
    "loss_mask",
    "position_ids",
    "hidden_states",
    "metadata_json",
)
_OPTIONAL_TENSOR_FIELDS = (
    "last_hidden_states",
    "target",
    "target_logprobs",
)


@dataclass(frozen=True)
class SampleMetadata:
    schema_version: int
    run_id: str
    sample_id: str
    sequence_no: int
    algorithm: str
    target_model_id: str
    target_model_revision: str
    tokenizer_fingerprint: str
    target_layer_ids: list[int]
    hidden_states_layout: str
    hidden_dtype: str
    hidden_shape: list[int]
    feature_length: int
    full_sequence_length: int
    feature_start: int
    feature_end: int
    use_logits: bool

    def validate(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported drafter sample schema_version={self.schema_version}; "
                f"expected {PROTOCOL_SCHEMA_VERSION}"
            )
        if not self.run_id:
            raise ValueError("SampleMetadata.run_id must not be empty")
        if not self.sample_id:
            raise ValueError("SampleMetadata.sample_id must not be empty")
        if self.sequence_no < 0:
            raise ValueError("SampleMetadata.sequence_no must be non-negative")
        if self.algorithm.strip().upper() != "DSPARK":
            raise ValueError(
                f"Standalone TQ protocol currently requires algorithm=DSPARK, got {self.algorithm!r}"
            )
        if len(self.hidden_shape) != 2:
            raise ValueError(
                f"SampleMetadata.hidden_shape must be [rows, hidden_dim], got {self.hidden_shape!r}"
            )
        if self.feature_length <= 0:
            raise ValueError("SampleMetadata.feature_length must be positive")
        if self.hidden_shape[0] != self.feature_length:
            raise ValueError(
                "SampleMetadata hidden_shape/feature_length mismatch: "
                f"{self.hidden_shape[0]} vs {self.feature_length}"
            )
        if not (0 <= self.feature_start < self.feature_end <= self.full_sequence_length):
            raise ValueError(
                "SampleMetadata feature window must satisfy "
                "0 <= feature_start < feature_end <= full_sequence_length"
            )
        if self.feature_end - self.feature_start != self.feature_length:
            raise ValueError(
                "SampleMetadata feature window length does not match feature_length: "
                f"{self.feature_end - self.feature_start} vs {self.feature_length}"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["algorithm"] = self.algorithm.strip().upper()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SampleMetadata":
        try:
            meta = cls(
                schema_version=int(payload["schema_version"]),
                run_id=str(payload["run_id"]),
                sample_id=str(payload["sample_id"]),
                sequence_no=int(payload["sequence_no"]),
                algorithm=str(payload["algorithm"]),
                target_model_id=str(payload["target_model_id"]),
                target_model_revision=str(payload["target_model_revision"]),
                tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
                target_layer_ids=[int(v) for v in payload["target_layer_ids"]],
                hidden_states_layout=str(payload["hidden_states_layout"]),
                hidden_dtype=str(payload["hidden_dtype"]),
                hidden_shape=[int(v) for v in payload["hidden_shape"]],
                feature_length=int(payload["feature_length"]),
                full_sequence_length=int(payload["full_sequence_length"]),
                feature_start=int(payload["feature_start"]),
                feature_end=int(payload["feature_end"]),
                use_logits=bool(payload["use_logits"]),
            )
        except KeyError as exc:
            raise ValueError(f"metadata_json missing required field {exc.args[0]!r}") from exc
        meta.validate()
        return meta


@dataclass(frozen=True)
class ExpectedFeatureConfig:
    """Consumer-side contract.  ``None`` fields are intentionally unchecked."""

    run_id: str
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    algorithm: str = "DSPARK"
    target_model_id: str | None = None
    target_model_revision: str | None = None
    tokenizer_fingerprint: str | None = None
    target_layer_ids: list[int] | None = None
    hidden_states_layout: str | None = None
    hidden_dtype: str | None = None


def make_sample_key(meta: SampleMetadata) -> str:
    meta.validate()
    return (
        f"drafter:v{meta.schema_version}:{meta.run_id}:"
        f"{meta.sequence_no:012d}:{meta.sample_id}"
    )


def make_ready_tag(meta: SampleMetadata) -> dict[str, Any]:
    meta.validate()
    return {
        "record_type": "sample",
        "status": "ready",
        "schema_version": meta.schema_version,
        "run_id": meta.run_id,
        "sequence_no": meta.sequence_no,
        "sample_id": meta.sample_id,
        "algorithm": meta.algorithm.strip().upper(),
    }


def encode_sample(
    sample: DraftFeatureSample | Mapping[str, Any], meta: SampleMetadata
) -> dict[str, torch.Tensor]:
    """Encode one normalized feature sample into TQ tensor fields."""

    meta.validate()
    normalized = (
        sample
        if isinstance(sample, DraftFeatureSample)
        else DraftFeatureSample.from_dict(dict(sample), strict=True)
    )
    normalized.validate(strict=True)
    if isinstance(normalized.hidden_states, (list, tuple)):
        raise TypeError("TQ drafter protocol requires hidden_states to be one dense tensor")
    hidden = _cpu_contiguous(normalized.hidden_states)
    input_ids = _cpu_contiguous(normalized.input_ids, dtype=torch.int64).reshape(-1)
    loss_mask = _cpu_contiguous(normalized.loss_mask, dtype=torch.float32).reshape(-1)
    if normalized.position_ids is None:
        position_ids = torch.arange(input_ids.numel(), dtype=torch.int64)
    else:
        position_ids = _cpu_contiguous(normalized.position_ids, dtype=torch.int64).reshape(-1)

    _validate_primary_tensors(input_ids, loss_mask, position_ids, hidden, meta)
    metadata_json = _json_to_tensor(meta.to_dict())
    fields: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "hidden_states": hidden,
        "metadata_json": metadata_json,
    }
    for field_name in _OPTIONAL_TENSOR_FIELDS:
        value = getattr(normalized, field_name)
        if value is not None:
            fields[field_name] = _cpu_contiguous(value)
    return fields


def decode_sample(
    key: str,
    tag: Mapping[str, Any],
    fields: Mapping[str, Any],
    expected_config: ExpectedFeatureConfig | Mapping[str, Any],
) -> DraftFeatureSample:
    """Validate a TQ record and restore the existing training sample type."""

    expected = (
        expected_config
        if isinstance(expected_config, ExpectedFeatureConfig)
        else ExpectedFeatureConfig(**dict(expected_config))
    )
    missing = [name for name in _REQUIRED_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"TQ sample {key!r} missing required fields: {missing}")
    metadata = SampleMetadata.from_dict(_tensor_to_json(fields["metadata_json"]))
    expected_key = make_sample_key(metadata)
    if key != expected_key:
        raise ValueError(f"TQ sample key mismatch: got {key!r}, expected {expected_key!r}")
    _validate_tag(tag, metadata)
    _validate_expected(metadata, expected)

    input_ids = _require_tensor(fields, "input_ids").detach().cpu().to(torch.int64).reshape(-1)
    loss_mask = _require_tensor(fields, "loss_mask").detach().cpu().to(torch.float32).reshape(-1)
    position_ids = _require_tensor(fields, "position_ids").detach().cpu().to(torch.int64).reshape(-1)
    hidden = _require_tensor(fields, "hidden_states").detach().cpu().contiguous()
    _validate_primary_tensors(input_ids, loss_mask, position_ids, hidden, metadata)

    payload: dict[str, Any] = {
        "schema_version": metadata.schema_version,
        "algorithm": metadata.algorithm,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "hidden_states": hidden,
        "metadata": metadata.to_dict(),
    }
    for field_name in _OPTIONAL_TENSOR_FIELDS:
        if field_name in fields and fields[field_name] is not None:
            payload[field_name] = _require_tensor(fields, field_name).detach().cpu().contiguous()
    return DraftFeatureSample.from_dict(payload, strict=True)


def make_eos_record(
    run_id: str, total_samples: int
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    if not run_id:
        raise ValueError("run_id must not be empty")
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    key = f"control:v{PROTOCOL_SCHEMA_VERSION}:{run_id}:eos"
    fields = {"marker": torch.tensor([1], dtype=torch.uint8)}
    tag = {
        "record_type": "control",
        "status": "eos",
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "run_id": run_id,
        "total_samples": int(total_samples),
    }
    return key, fields, tag


def _validate_tag(tag: Mapping[str, Any], meta: SampleMetadata) -> None:
    expected = make_ready_tag(meta)
    for name, expected_value in expected.items():
        if tag.get(name) != expected_value:
            raise ValueError(
                f"TQ sample tag mismatch for {name}: got {tag.get(name)!r}, "
                f"expected {expected_value!r}"
            )


def _validate_expected(meta: SampleMetadata, expected: ExpectedFeatureConfig) -> None:
    checks = {
        "run_id": expected.run_id,
        "schema_version": expected.schema_version,
        "algorithm": expected.algorithm.strip().upper(),
        "target_model_id": expected.target_model_id,
        "target_model_revision": expected.target_model_revision,
        "tokenizer_fingerprint": expected.tokenizer_fingerprint,
        "target_layer_ids": expected.target_layer_ids,
        "hidden_states_layout": expected.hidden_states_layout,
        "hidden_dtype": expected.hidden_dtype,
    }
    for name, expected_value in checks.items():
        if expected_value is None:
            continue
        actual = getattr(meta, name)
        if name == "algorithm":
            actual = str(actual).strip().upper()
        if actual != expected_value:
            raise ValueError(
                f"TQ sample metadata mismatch for {name}: got {actual!r}, "
                f"expected {expected_value!r}"
            )


def _validate_primary_tensors(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    hidden: torch.Tensor,
    meta: SampleMetadata,
) -> None:
    if hidden.dim() == 3 and hidden.size(0) == 1:
        hidden = hidden.squeeze(0)
    if hidden.dim() != 2:
        raise ValueError(f"hidden_states must have shape [L,D], got {tuple(hidden.shape)}")
    lengths = {
        "input_ids": int(input_ids.numel()),
        "loss_mask": int(loss_mask.numel()),
        "position_ids": int(position_ids.numel()),
        "hidden_states": int(hidden.size(0)),
    }
    if any(value != meta.feature_length for value in lengths.values()):
        raise ValueError(
            f"TQ sample tensor lengths must equal feature_length={meta.feature_length}: {lengths}"
        )
    if list(hidden.shape) != meta.hidden_shape:
        raise ValueError(
            f"hidden_states shape mismatch: got {list(hidden.shape)}, expected {meta.hidden_shape}"
        )
    actual_dtype = _dtype_name(hidden.dtype)
    if actual_dtype != meta.hidden_dtype:
        raise ValueError(
            f"hidden_states dtype mismatch: got {actual_dtype!r}, expected {meta.hidden_dtype!r}"
        )


def _json_to_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _tensor_to_json(value: Any) -> dict[str, Any]:
    tensor = value
    if not torch.is_tensor(tensor):
        raise TypeError("metadata_json must be a torch.Tensor")
    tensor = tensor.detach().cpu().to(torch.uint8).reshape(-1)
    try:
        decoded = json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    return decoded


def _require_tensor(fields: Mapping[str, Any], name: str) -> torch.Tensor:
    value = fields.get(name)
    if not torch.is_tensor(value):
        raise TypeError(f"TQ field {name!r} must be a torch.Tensor")
    return value


def _cpu_contiguous(value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected torch.Tensor, got {type(value)!r}")
    result = value.detach().cpu()
    if dtype is not None:
        result = result.to(dtype)
    return result.contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


__all__ = [
    "DRAFTER_TQ_PARTITION",
    "PROTOCOL_SCHEMA_VERSION",
    "ExpectedFeatureConfig",
    "SampleMetadata",
    "decode_sample",
    "encode_sample",
    "make_eos_record",
    "make_ready_tag",
    "make_sample_key",
]
