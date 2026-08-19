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

import pytest

torch = pytest.importorskip("torch")

from verl_speco.trainer.feature_store import DraftFeatureSample  # noqa: E402
from verl_speco.trainer.target_feature_replay import (  # noqa: E402
    BoundedReplayCache,
    _hidden_capture_target,
)


def _feature_sample() -> DraftFeatureSample:
    return DraftFeatureSample(
        input_ids=torch.arange(8),
        loss_mask=torch.ones(8),
        hidden_states=torch.zeros(8, 16),
        position_ids=torch.arange(1, 9),
    )


def test_hidden_capture_target_matches_transformers_hidden_state_indices():
    assert _hidden_capture_target(0, 36) == ("layer", 0)
    assert _hidden_capture_target(34, 36) == ("layer", 34)
    assert _hidden_capture_target(35, 36) == ("final", None)


def test_bounded_replay_cache_roundtrip(tmp_path):
    cache = BoundedReplayCache(
        tmp_path,
        max_size_gb=0.01,
        rank=1,
        world_size=2,
    )

    assert cache.put("sample", _feature_sample()) is True
    loaded = cache.get("sample")

    assert loaded is not None
    assert torch.equal(loaded.input_ids, torch.arange(8))
    assert cache.metrics()["replay/cache_size_gb"] > 0
    assert (tmp_path / "rank00001" / "sample.pt").exists()


def test_bounded_replay_cache_disables_zero_budget(tmp_path):
    cache = BoundedReplayCache(
        tmp_path,
        max_size_gb=0,
        rank=0,
        world_size=1,
    )

    assert cache.put("sample", _feature_sample()) is False
    assert cache.get("sample") is None
