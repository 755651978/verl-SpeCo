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

import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from verl_speco.trainer.standalone_checkpoint import rewrite_standalone_runtime_config  # noqa: E402
from verl_speco.trainer.draft_training_loop import (  # noqa: E402
    _build_backend,
    _is_out_of_memory_error,
    _rewrite_standalone_block_runtime_config,
    _save_standalone_checkpoint,
    _torch_load_cpu,
)


class _FakeTrainer:
    def __init__(self):
        self.checkpoint_dir = "/tmp/draft"
        self._pending_full_checkpoint_future = None
        self.future = Future()
        self.calls = 0

    def _save_checkpoint_async(self, step: int):
        self.calls += 1
        self.step = step
        self._pending_full_checkpoint_future = self.future
        return self.future


@pytest.mark.parametrize(
    ("attempted_batches", "expected"),
    [
        (1, True),
        (2, True),
        (3, True),
        (4, False),
        (99, False),
        (100, True),
        (101, False),
    ],
)
def test_should_log_standalone_batch_progress(attempted_batches, expected):
    assert _should_log_batch_progress(attempted_batches) is expected


def test_is_out_of_memory_error_matches_npu_oom_message():
    error = RuntimeError("NPU out of memory. Tried to allocate 258.00 MiB")

    assert _is_out_of_memory_error(error)
    assert not _is_out_of_memory_error(RuntimeError("bad batch"))


def _export_trainer(model_type: str, model_path=None):
    """Minimal trainer stand-in for the standalone checkpoint export helpers."""
    return SimpleNamespace(
        backend=SimpleNamespace(model_type=model_type),
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=model_path))
        ),
    )


def _standalone_config(algorithm: str):
    return OmegaConf.create(
        {
            "model": {"path": "/does/not/exist"},
            "rollout": {
                "drafter": {"speculative_algorithm": algorithm, "training": {}}
            },
        }
    )


@pytest.mark.parametrize(
    ("algorithm", "expected_backend", "expected_model_type"),
    [
        ("EAGLE3", "Eagle3TrainerBackend", "eagle3"),
        ("EAGLE1", "Eagle1TrainerBackend", "eagle3"),
        ("eagle2", "Eagle1TrainerBackend", "eagle3"),
        ("DFLASH", "DFlashTrainerBackend", "dflash"),
        ("DSPARK", "DSparkTrainerBackend", "dspark"),
        ("DOMINO", "DominoTrainerBackend", "domino"),
        ("PEAGLE", "PEagleTrainerBackend", "peagle"),
    ],
)
def test_standalone_backend_covers_every_online_algorithm(
    algorithm, expected_backend, expected_model_type
):
    backend = _build_backend(_standalone_config(algorithm))

    assert type(backend).__name__ == expected_backend
    assert backend.model_type == expected_model_type


def test_standalone_backend_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="Unsupported drafter algorithm"):
        _build_backend(_standalone_config("NOT_AN_ALGORITHM"))


def test_standalone_checkpoint_schedules_without_waiting():
    trainer = _FakeTrainer()

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is True
    assert result["reason"] == "scheduled"
    assert trainer.calls == 1
    assert trainer._pending_full_checkpoint_future is trainer.future


def test_standalone_checkpoint_waits_when_requested():
    trainer = _FakeTrainer()
    trainer.future.set_result(None)

    result = _save_standalone_checkpoint(trainer, 5, wait=True)

    assert result["saved"] is True
    assert result["reason"] == "saved"
    assert trainer._pending_full_checkpoint_future is None


def test_standalone_checkpoint_skips_when_previous_save_is_running():
    trainer = SimpleNamespace(
        checkpoint_dir="/tmp/draft", _pending_full_checkpoint_future=Future()
    )

    result = _save_standalone_checkpoint(trainer, 5)

    assert result["saved"] is False
    assert result["reason"] == "previous_save_running"


def test_public_checkpoint_path_rewrites_dspark_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepSeekDSparkModel"],
                "target_layer_ids": [1, 9, 17],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dspark",
                "architectures": ["DSparkDraftModel"],
                "target_layer_ids": [1, 9, 17],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )

    class _PublicCheckpointTrainer:
        backend = SimpleNamespace(model_type="dspark")
        config = SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        )

        @staticmethod
        def save_checkpoint(step: int, wait: bool):
            assert step == 5
            assert wait is True
            return {"saved": True, "reason": "saved", "path": str(checkpoint_dir)}

    result = _save_standalone_checkpoint(_PublicCheckpointTrainer(), 5, wait=True)

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert result["saved"] is True
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert (checkpoint_dir / "speco_training_config.json").exists()


def test_standalone_checkpoint_rewrites_runtime_config_after_save(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {"model_type": "deepseek_v3", "architectures": ["DeepSeekDSparkModel"]}
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    events = []

    class _CheckpointTrainer:
        backend = SimpleNamespace(model_type="dspark")
        config = SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        )

        @staticmethod
        def save_checkpoint(step: int, wait: bool):
            assert step == 5
            assert wait is True
            events.append("save")
            return {"saved": True, "reason": "saved", "path": str(checkpoint_dir)}

    result = _save_standalone_checkpoint(_CheckpointTrainer(), 5, wait=True)

    assert result["saved"] is True
    assert events == ["save"]


def test_standalone_dspark_checkpoint_preserves_source_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    source_config = {
        "model_type": "deepseek_v3",
        "architectures": ["DeepSeekDSparkModel"],
        "target_layer_ids": [1, 9, 17],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "dspark",
        "architectures": ["DSparkDraftModel"],
        "target_layer_ids": [1, 9, 17],
        "mask_token_id": 151669,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "block_size": 7,
        "num_context_layers": 3,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = _export_trainer("dspark", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "deepseek_v3"
    assert runtime_config["architectures"] == ["DeepSeekDSparkModel"]
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert runtime_config["dflash_config"]["target_layer_ids"] == [1, 9, 17]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18]
    assert saved_training_config == training_config


def test_standalone_dspark_checkpoint_rewrites_generic_qwen3_architecture(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DSparkDraftModel"],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}), encoding="utf-8"
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dspark",
                "architectures": ["DSparkDraftModel"],
                "markov_head_type": "vanilla",
            }
        ),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(source_dir))
            ),
        ),
    )

    rewrite_standalone_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["Qwen3DSparkModel"]
    assert runtime_config["speco_training_model_type"] == "dspark"


def test_standalone_domino_checkpoint_exports_dflash_projector_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_domino"
    source_dir.mkdir()
    source_config = {
        "model_type": "qwen3",
        "architectures": ["DominoDraftModel"],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "domino",
        "architectures": ["DominoDraftModel"],
        "target_layer_ids": [2, 10, 18],
        "mask_token_id": 151669,
        "num_context_layers": 3,
        "block_size": 16,
        "num_anchors": 512,
        "projector_type": "domino",
        "emb_dim": 256,
        "gru_hidden_dim": 1024,
        "pure_draft_prefix_len": 1,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = _export_trainer("domino", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    dflash_config = runtime_config["dflash_config"]
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["speco_training_model_type"] == "domino"
    # Engines serve Domino through the DFlash method and switch on projector_type.
    assert dflash_config["projector_type"] == "domino"
    assert dflash_config["emb_dim"] == 256
    assert dflash_config["gru_hidden_dim"] == 1024
    assert dflash_config["pure_draft_prefix_len"] == 1
    assert dflash_config["block_size"] == 16
    assert dflash_config["target_layer_ids"] == [2, 10, 18]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [3, 11, 19]
    assert saved_training_config == training_config


def test_standalone_domino_checkpoint_defaults_projector_type(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "domino", "architectures": ["DominoDraftModel"]}),
        encoding="utf-8",
    )
    trainer = _export_trainer("domino", None)

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["dflash_config"]["projector_type"] == "domino"


def test_standalone_dflash_checkpoint_preserves_source_runtime_config(tmp_path):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dflash"
    source_dir.mkdir()
    source_config = {
        "model_type": "qwen3",
        "architectures": ["DFlashForCausalLM"],
    }
    (source_dir / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    training_config = {
        "model_type": "dflash",
        "architectures": ["DFlashDraftModel"],
        "target_layer_ids": [2, 10, 18],
        "mask_token_id": 151669,
        "num_context_layers": 3,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = _export_trainer("dflash", str(source_dir))

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["DFlashForCausalLM"]
    assert runtime_config["dflash_config"]["target_layer_ids"] == [2, 10, 18]
    assert runtime_config["eagle_aux_hidden_state_layer_ids"] == [3, 11, 19]
    assert saved_training_config == training_config


def test_standalone_block_checkpoint_uses_target_model_type_without_source_config(
    tmp_path,
):
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_dspark"
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "head_dim": 128,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    training_config = {
        "model_type": "dspark",
        "architectures": ["DSparkDraftModel"],
        "target_layer_ids": [1, 9, 17],
        "markov_head_type": "vanilla",
        "head_dim": 80,
        "rope_theta": 10000.0,
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    source_model_path = rewrite_standalone_runtime_config(trainer, str(checkpoint_dir))

    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    saved_training_config = json.loads(
        (checkpoint_dir / "speco_training_config.json").read_text(encoding="utf-8")
    )
    assert source_model_path == str(missing_source_dir)
    assert runtime_config["model_type"] == "qwen3"
    assert runtime_config["architectures"] == ["Qwen3DSparkModel"]
    assert runtime_config["draft_model_type"] == "dspark"
    assert runtime_config["speculative_algorithm"] == "DSPARK"
    assert runtime_config["speco_training_model_type"] == "dspark"
    assert runtime_config["head_dim"] == 128
    assert runtime_config["max_position_embeddings"] == 40960
    assert runtime_config["dflash_config"]["head_dim"] == 128
    assert runtime_config["dspark_config"]["head_dim"] == 128
    assert runtime_config["dspark_config"]["markov_head_type"] == "vanilla"
    assert runtime_config["rope_parameters"] == {
        "rope_theta": 1000000.0,
        "rope_type": "default",
    }
    assert "rope_theta" not in runtime_config
    assert "rope_theta" not in runtime_config["dflash_config"]
    assert "rope_theta" not in runtime_config["dspark_config"]
    assert saved_training_config == training_config


def test_standalone_dflash_checkpoint_preserves_source_lm_head(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dflash"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["DFlashForCausalLM"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dflash", "architectures": ["DFlashDraftModel"]}),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {"lm_head.weight": torch.ones(3, 4)}, str(source_dir / "model.safetensors")
    )
    safetensors_torch.save_file(
        {"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors")
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dflash"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device="cpu"
    )
    assert torch.equal(exported_state["lm_head.weight"], torch.ones(3, 4))
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_dflash_checkpoint_does_not_create_lm_head_without_source(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_dflash"
    (target_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3"}), encoding="utf-8"
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dflash", "architectures": ["DFlashDraftModel"]}),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {"model.embed_tokens.weight": torch.ones(3, 4)},
        str(target_dir / "model.safetensors"),
    )
    safetensors_torch.save_file(
        {"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors")
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dflash"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device="cpu"
    )
    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert "lm_head.weight" not in exported_state
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_dspark_checkpoint_appends_target_tied_embedding(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_dspark"
    (target_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors_torch.save_file(
        {"model.embed_tokens.weight": embedding}, str(target_dir / "model.safetensors")
    )
    safetensors_torch.save_file(
        {"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors")
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device="cpu"
    )
    runtime_config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_type"] == "qwen3"
    assert torch.equal(exported_state["lm_head.weight"], embedding)
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_dspark_checkpoint_skips_untied_target_embedding(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    target_dir = tmp_path / "target_qwen3"
    target_dir.mkdir()
    missing_source_dir = tmp_path / "missing_source_dspark"
    (target_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "tie_word_embeddings": False}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    safetensors_torch.save_file(
        {"model.embed_tokens.weight": torch.ones(3, 4)},
        str(target_dir / "model.safetensors"),
    )
    safetensors_torch.save_file(
        {"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors")
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            model=SimpleNamespace(path=str(target_dir)),
            rollout=SimpleNamespace(
                drafter=SimpleNamespace(model_path=str(missing_source_dir))
            ),
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device="cpu"
    )
    assert "lm_head.weight" not in exported_state
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_block_checkpoint_appends_source_lm_head_weight(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["DSparkForCausalLM"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    lm_head = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    safetensors_torch.save_file(
        {"lm_head.weight": lm_head}, str(source_dir / "model.safetensors")
    )
    safetensors_torch.save_file(
        {"fc.weight": torch.ones(2, 2)}, str(checkpoint_dir / "model.safetensors")
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    exported_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device="cpu"
    )
    assert torch.equal(exported_state["lm_head.weight"], lm_head)
    assert torch.equal(exported_state["fc.weight"], torch.ones(2, 2))


def test_standalone_block_checkpoint_appends_lm_head_to_sharded_safetensors_index(
    tmp_path,
):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint_dir = tmp_path / "draft_step_5"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source_dspark"
    source_dir.mkdir()
    (source_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["DSparkForCausalLM"]}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model_type": "dspark", "architectures": ["DSparkDraftModel"]}),
        encoding="utf-8",
    )
    lm_head = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fc_weight = torch.ones(2, 2)
    safetensors_torch.save_file(
        {"lm_head.weight": lm_head}, str(source_dir / "model.safetensors")
    )
    safetensors_torch.save_file(
        {"fc.weight": fc_weight},
        str(checkpoint_dir / "model-00001-of-00001.safetensors"),
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": fc_weight.numel() * fc_weight.element_size()
                },
                "weight_map": {"fc.weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        backend=SimpleNamespace(model_type="dspark"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(drafter=SimpleNamespace(model_path=str(source_dir)))
        ),
    )

    _rewrite_standalone_block_runtime_config(trainer, str(checkpoint_dir))

    index_data = json.loads(
        (checkpoint_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index_data["weight_map"]["lm_head.weight"] == "model-lm-head.safetensors"
    assert index_data["metadata"]["total_size"] == (
        fc_weight.numel() * fc_weight.element_size()
        + lm_head.numel() * lm_head.element_size()
    )
    added_state = safetensors_torch.load_file(
        str(checkpoint_dir / "model-lm-head.safetensors"), device="cpu"
    )
    assert torch.equal(added_state["lm_head.weight"], lm_head)


def test_torch_load_cpu_falls_back_without_weights_only(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "pytorch_model.bin"
    expected = {"lm_head.weight": torch.ones(2, 2)}
    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("weights_only is unsupported")
        assert path == str(checkpoint_path)
        return expected

    monkeypatch.setattr(torch, "load", fake_load)

    assert _torch_load_cpu(str(checkpoint_path)) is expected
    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]
