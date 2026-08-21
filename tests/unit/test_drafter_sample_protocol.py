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
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.transport.drafter_sample_protocol import (
    ExpectedFeatureConfig,
    SampleMetadata,
    decode_sample,
    encode_sample,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
)


def _metadata() -> SampleMetadata:
    return SampleMetadata(
        schema_version=1,
        run_id="run-a",
        sample_id="train-000017",
        sequence_no=17,
        algorithm="DSPARK",
        target_model_id="/models/Qwen3-8B",
        target_model_revision="rev-a",
        tokenizer_fingerprint="sha256:tokenizer",
        target_layer_ids=[2, 8, 14, -1],
        hidden_states_layout="dflash_aux_plus_last",
        hidden_dtype="bfloat16",
        hidden_shape=[4, 16],
        feature_length=4,
        full_sequence_length=10,
        feature_start=6,
        feature_end=10,
        use_logits=False,
    )


def _sample() -> DraftFeatureSample:
    return DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=torch.tensor([10, 11, 12, 13]),
        loss_mask=torch.tensor([0.0, 1.0, 1.0, 1.0]),
        position_ids=torch.tensor([6, 7, 8, 9]),
        hidden_states=torch.arange(64, dtype=torch.bfloat16).reshape(4, 16),
        metadata={"ignored_at_wire_boundary": True},
    )


def _expected() -> ExpectedFeatureConfig:
    meta = _metadata()
    return ExpectedFeatureConfig(
        run_id=meta.run_id,
        target_model_id=meta.target_model_id,
        target_model_revision=meta.target_model_revision,
        tokenizer_fingerprint=meta.tokenizer_fingerprint,
        target_layer_ids=meta.target_layer_ids,
        hidden_states_layout=meta.hidden_states_layout,
        hidden_dtype=meta.hidden_dtype,
    )


def test_sample_round_trip() -> None:
    meta = _metadata()
    key = make_sample_key(meta)
    fields = encode_sample(_sample(), meta)
    restored = decode_sample(key, make_ready_tag(meta), fields, _expected())

    assert key == "drafter:v1:run-a:000000000017:train-000017"
    assert tuple(restored.hidden_states.shape) == (4, 16)
    assert restored.hidden_states.dtype == torch.bfloat16
    assert restored.input_ids.tolist() == [10, 11, 12, 13]
    assert restored.metadata["sequence_no"] == 17
    assert restored.metadata["hidden_states_layout"] == "dflash_aux_plus_last"
    assert fields["metadata_json"].dtype == torch.uint8


def test_decode_rejects_identity_mismatch() -> None:
    meta = _metadata()
    fields = encode_sample(_sample(), meta)
    bad_tag = {**make_ready_tag(meta), "sample_id": "wrong"}
    with pytest.raises(ValueError, match="tag mismatch for sample_id"):
        decode_sample(make_sample_key(meta), bad_tag, fields, _expected())


def test_decode_rejects_consumer_contract_mismatch() -> None:
    meta = _metadata()
    fields = encode_sample(_sample(), meta)
    expected = replace(_expected(), target_model_revision="different")
    with pytest.raises(ValueError, match="target_model_revision"):
        decode_sample(make_sample_key(meta), make_ready_tag(meta), fields, expected)


def test_encode_rejects_shape_mismatch() -> None:
    meta = replace(_metadata(), hidden_shape=[4, 32])
    with pytest.raises(ValueError, match="hidden_states shape mismatch"):
        encode_sample(_sample(), meta)


def test_protocol_algorithm_is_not_hardcoded_to_dspark() -> None:
    meta = replace(_metadata(), algorithm="EAGLE3")
    sample = replace(_sample(), algorithm="EAGLE3")
    expected = replace(_expected(), algorithm="EAGLE3")

    key = make_sample_key(meta)
    fields = encode_sample(sample, meta)
    restored = decode_sample(key, make_ready_tag(meta), fields, expected)

    assert make_ready_tag(meta)["algorithm"] == "EAGLE3"
    assert restored.algorithm == "EAGLE3"


def test_eos_record_is_control_only() -> None:
    key, fields, tag = make_eos_record("run-a", 18)
    assert key == "control:v1:run-a:eos"
    assert fields["marker"].tolist() == [1]
    assert tag == {
        "record_type": "control",
        "status": "eos",
        "schema_version": 1,
        "run_id": "run-a",
        "total_samples": 18,
    }
