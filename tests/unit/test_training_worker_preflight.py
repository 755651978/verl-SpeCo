# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("torch")

from omegaconf import OmegaConf

from verl_speco.trainer.base_trainer import DrafterBaseTrainer
from verl_speco.workers.speco_worker import SpecoWorker


class _FakeTrainer:
    def __init__(self, *, data_version: int) -> None:
        self.buffer_version = 3
        self._target_lm_head_weight_step = 4
        self.optimizer_steps_total = 0
        self.data_version = data_version
        self.activation_calls = 0
        self.cleanup_calls = 0
        self.release_after_activation_calls = 0
        self.reserved_plan_id = None

    def select_target_lm_head_version(self, global_step: int) -> bool:
        self._target_lm_head_weight_step = global_step
        return True

    def reserve_training_data(self, *, plan_id: str, **kwargs):
        del kwargs
        self.reserved_plan_id = plan_id
        return {"reserved_samples": 4}

    def release_training_data_reservation(self, plan_id: str) -> int:
        if self.reserved_plan_id != plan_id:
            return 0
        self.reserved_plan_id = None
        return 4

    def get_training_data_status(self, **kwargs):
        del kwargs
        return {
            "trainable_batches": 1,
            "trainable_samples": 4,
            "data_version": self.data_version,
        }

    async def activate_training_model(self) -> bool:
        self.activation_calls += 1
        return True

    async def cleanup_training(self, clear_data: bool = True):
        del clear_data
        self.cleanup_calls += 1

    async def release_training_memory_after_activation(self):
        self.release_after_activation_calls += 1

    def pop_model_state_dict_for_publish(self, global_step: int):
        return global_step == 4, {"weight": global_step}


def _worker(*, data_version: int) -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker.rank = 0
    worker.worker_incarnation = "worker-0"
    worker.last_global_step = 4
    worker.device_name = "cpu"
    worker.trainer = _FakeTrainer(data_version=data_version)
    worker._prepared_training_plan_id = None
    worker._prepared_training_data_version = None
    worker._prepared_training_target_version = None
    worker.replica_rank = 0
    worker.last_trained_step = None
    worker.is_drafter_group_leader = True
    worker.is_global_publish_leader = True
    worker._last_trained_execution_strategy = None
    worker._last_trained_target_worker_ids = ()
    worker.training_group_ranks = [0]
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "drafter": {
                    "training": {
                        "scheduler": {
                            "execution": {"strategy": "rollout_idle_worker"}
                        }
                    }
                }
            }
        }
    )
    return worker


def _plan() -> dict[str, object]:
    return {
        "launch": True,
        "execution_strategy": "sync",
        "source_global_step": 4,
        "plan_id": "plan-4",
        "data_version": 4,
        "required_target_version": 4,
        "sample_last_n_steps": 2,
        "require_full_batch": False,
        "min_batches": 1,
        "max_batches": 1,
        "worker_snapshots": {
            "0": {
                "worker_incarnation": "worker-0",
                "buffer_version": 3,
                "data_version": 4,
            }
        },
    }


def test_worker_preflight_rejects_changed_data_version_before_activation() -> None:
    worker = _worker(data_version=5)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "data_version_changed"
    assert result["data_version"] == 5
    assert worker.trainer.activation_calls == 0
    assert worker._prepared_training_plan_id is None


def test_worker_preflight_records_actual_versions_for_training_result() -> None:
    worker = _worker(data_version=4)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert result["ready"]
    assert result["data_version"] == 4
    assert result["target_version"] == 4
    assert worker._prepared_training_plan_id == "plan-4"
    assert worker._prepared_training_data_version == 4
    assert worker._prepared_training_target_version == 4


def test_worker_idle_prewarm_keeps_training_model_hot() -> None:
    worker = _worker(data_version=4)

    result = asyncio.run(worker.prewarm_drafter_training_model())

    assert result["activated"]
    assert result["reason"] == "prewarmed"
    assert worker.trainer.activation_calls == 1
    assert worker.trainer.release_after_activation_calls == 0
    assert worker.trainer.cleanup_calls == 0


def test_bubble_publish_uses_replica_local_group_leader(monkeypatch) -> None:
    worker = _worker(data_version=4)
    worker.last_trained_step = 4
    worker.is_global_publish_leader = False
    worker.is_drafter_group_leader = True
    worker._last_trained_execution_strategy = "rollout_idle_worker"
    worker._last_trained_target_worker_ids = ("2", "3")
    monkeypatch.setattr("verl_speco.workers.speco_worker.ray.put", lambda value: value)

    result = worker.maybe_publish()

    assert result == {"weights_ref": {"weight": 4}}


def test_sync_publish_still_uses_global_leader(monkeypatch) -> None:
    worker = _worker(data_version=4)
    worker.last_trained_step = 4
    worker.is_global_publish_leader = False
    worker.is_drafter_group_leader = True
    worker._last_trained_execution_strategy = "sync"
    monkeypatch.setattr("verl_speco.workers.speco_worker.ray.put", lambda value: value)

    assert worker.maybe_publish() is None


def test_replica_local_idle_trainer_config_enables_use_orig_params() -> None:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.rank = 2
    worker.config = OmegaConf.create(
        {
            "actor": {
                "fsdp_config": {
                    "use_orig_params": False,
                    "forward_prefetch": False,
                }
            },
            "rollout": {
                "drafter": {
                    "training": {
                        "fsdp_config": {
                            "use_orig_params": False,
                            "forward_prefetch": True,
                        }
                    }
                }
            },
        }
    )

    trainer_config = worker._replica_local_idle_trainer_config()

    assert trainer_config.actor.fsdp_config.use_orig_params is True
    assert (
        trainer_config.rollout.drafter.training.fsdp_config.use_orig_params is True
    )
    assert worker.config.actor.fsdp_config.use_orig_params is False
    assert worker.config.rollout.drafter.training.fsdp_config.use_orig_params is False


def test_base_trainer_orig_params_override_is_replica_local_bubble_only() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer._bubble_time_enabled = True
    trainer.training_device_mesh = None
    trainer.training_process_group = object()
    trainer.training_group_world_size = 2

    assert trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer._bubble_time_enabled = False
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer._bubble_time_enabled = True
    trainer.training_device_mesh = object()
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()

    trainer.training_device_mesh = None
    trainer.training_process_group = None
    assert not trainer._replica_local_bubble_fsdp_requires_orig_params()
