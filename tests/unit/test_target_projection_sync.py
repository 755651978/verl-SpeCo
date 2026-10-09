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

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_methods(relative_path, class_name, names, namespace):
    """Execute production method bodies without importing the GPU/Ray stack."""
    path = Path(__file__).resolve().parents[2] / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
    for node in methods:
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    loaded_class = type(class_name, (), {name: namespace[name] for name in names})
    namespace[class_name] = loaded_class
    return loaded_class


@pytest.fixture
def trainer():
    cls = _load_methods(
        "verl_speco/trainer/speco_ray_trainer.py",
        "SpecoRayPPOTrainer",
        [
            "_speco_owner_route_mapping",
            "_speco_build_drafter_target_lm_head_sync_args",
            "_speco_target_sync_worker_replicas",
            "_speco_start_target_lm_head_weight_sync",
            "_flatten_target_sync_results",
            "_speco_finish_target_lm_head_weight_sync",
        ],
        {
            "time": time,
            "_DRAFTER_TARGET_SYNC_MESH": "drafter_target_sync",
            "_get_nested": lambda cfg, keys, default: default,
        },
    )
    instance = cls()
    # The production method is static; retain that binding after extraction.
    cls._flatten_target_sync_results = staticmethod(cls._flatten_target_sync_results)
    instance._ray_get_if_needed = lambda value: value
    instance._first_non_null = lambda values: next(
        (value for value in values if value is not None), None
    )
    instance.global_steps = 5
    instance.config = {}
    instance.device_name = "npu"
    instance.drafter_wg = SimpleNamespace(
        world_size=4,
        _dispatch_info={
            "drafter_target_sync": [0, 0, 0, 0],
            "drafter_owner_route": [0, 0, 1, 1],
        },
        _collect_info={"drafter_target_sync": [True] * 4},
    )
    instance._speco_drafter_training_config = lambda: {}
    instance._speco_get_drafter_target_lm_head_row_selection = lambda: None
    return instance


def _acks():
    return [
        {
            "accepted": True,
            "pending": True,
            "global_step": 5,
            "worker_id": str(rank),
            "replica_rank": rank // 2,
            "projection_fingerprint": "verified",
        }
        for rank in range(4)
    ]


def _start_sync(trainer, results, fingerprint="verified"):
    payload = {"weight": object(), "projection_fingerprint": fingerprint}
    trainer._speco_actor_rollout_method = lambda name: lambda *args, **kwargs: [payload]
    trainer.speco_sync_target_lm_head_weight = lambda *args, **kwargs: results
    _, pending = trainer._speco_start_target_lm_head_weight_sync()
    return pending


def test_one_broadcast_bucket_accepts_four_workers_in_two_replicas(trainer):
    payload_args, steps, bucket_count = (
        trainer._speco_build_drafter_target_lm_head_sync_args({})
    )
    assert len(payload_args) == bucket_count == 1
    assert steps == [5]
    pending = _start_sync(trainer, _acks())
    assert pending["expected_results"] == 4
    assert pending["expected_worker_replicas"] == {"0": 0, "1": 0, "2": 1, "3": 1}
    assert (
        trainer._speco_finish_target_lm_head_weight_sync(pending)[
            "drafter/target_lm_head_synced"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "extra",
        "replica",
        "step",
        "fingerprint",
        "rejected",
        "unstaged",
    ],
)
def test_sync_rejects_invalid_worker_acknowledgements(trainer, failure):
    results = _acks()
    if failure == "missing":
        results.pop()
    elif failure == "duplicate":
        results[-1] = dict(results[0])
    elif failure == "extra":
        results.append(dict(results[0], worker_id="4"))
    else:
        key, value = {
            "replica": ("replica_rank", 7),
            "step": ("global_step", 4),
            "fingerprint": ("projection_fingerprint", "wrong"),
            "rejected": ("accepted", False),
            "unstaged": ("pending", False),
        }[failure]
        results[0][key] = value
    with pytest.raises(RuntimeError, match="worker acknowledgement failed"):
        trainer._speco_finish_target_lm_head_weight_sync(_start_sync(trainer, results))


def test_non_peft_sync_accepts_acknowledgements_without_fingerprints(trainer):
    results = _acks()
    for result in results:
        result.pop("projection_fingerprint")
    assert (
        trainer._speco_finish_target_lm_head_weight_sync(
            _start_sync(trainer, results, fingerprint=None)
        )["drafter/target_lm_head_synced"]
        == 1
    )


def test_topology_uses_target_sync_collect_mask(trainer):
    trainer.drafter_wg._collect_info["drafter_target_sync"] = [True, False, True, False]
    assert trainer._speco_target_sync_worker_replicas() == {"0": 0, "2": 1}


def test_topology_queries_and_caches_missing_metadata(trainer):
    queries = []
    trainer.drafter_wg._dispatch_info.pop("drafter_owner_route")
    trainer.drafter_wg._collect_info.clear()
    trainer.drafter_wg._query_dispatch_info = lambda mesh: queries.append(mesh) or [
        0,
        0,
        1,
        1,
    ]
    trainer.drafter_wg._query_collect_info = (
        lambda mesh: queries.append(mesh) or [True] * 4
    )
    for _ in range(2):
        assert trainer._speco_target_sync_worker_replicas() == {
            "0": 0,
            "1": 0,
            "2": 1,
            "3": 1,
        }
    assert queries == ["drafter_owner_route", "drafter_target_sync"]


@pytest.mark.parametrize("fingerprint", [None, "verified", "wrong"])
def test_worker_hashes_only_payloads_carrying_fingerprints(fingerprint):
    calls = []
    cls = _load_methods(
        "verl_speco/workers/speco_worker.py",
        "SpecoWorker",
        ["sync_target_lm_head_weight"],
        {
            "_projection_fingerprint": lambda payload: calls.append(payload)
            or "verified"
        },
    )
    worker = cls()
    worker.enable_drafter = worker.in_drafter_train_group = True
    worker.rank = worker.replica_rank = 0
    worker.worker_incarnation = "test"
    worker.is_drafter_group_leader = False
    applied = []
    worker.trainer = SimpleNamespace(
        sync_target_lm_head_weight=lambda weight, **kwargs: applied.append(kwargs)
        or {"accepted": True}
    )
    result = worker.sync_target_lm_head_weight(
        {"weight": object(), "projection_fingerprint": fingerprint}, global_step=5
    )
    assert len(calls) == int(fingerprint is not None)
    assert result["accepted"] is (fingerprint != "wrong")
    assert len(applied) == int(fingerprint != "wrong")
