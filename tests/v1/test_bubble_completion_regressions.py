# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Regressions against real V1 hooks and native checkpoint ordering."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle
from verl_speco.trainer.scheduler import DrafterScheduler
from verl_speco.trainer.v1.factory import _trainer_types

try:
    TRAINERS = _trainer_types()
except (ImportError, RuntimeError):
    if os.getenv("SPECO_REQUIRE_V1") == "1":
        raise
    pytest.skip("requires upstream verl 0.9 and TransferQueue", allow_module_level=True)


def config(mode="sync"):
    return OmegaConf.create(
        {
            "trainer": {"device": "cpu", "v1": {"trainer_mode": mode}},
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": False}},
                "rollout": {
                    "data_parallel_size": 1,
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "scheduler": {
                                "execution": {"strategy": "rollout_idle_worker"}
                            }
                        },
                    },
                },
            },
        }
    )


def trainer_for(mode):
    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = 7
    trainer.trainer_mode = mode
    trainer._drafter_scheduler = DrafterScheduler()
    trainer._speco_bubble_lifecycle = BubbleLifecycle(trainer)
    return trainer


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_generation_start_uses_authoritative_replica_groups(mode):
    trainer = trainer_for(mode)
    scheduler = trainer._drafter_scheduler
    scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": replica,
                "training_group_ranks": ranks,
                "full_collective_ranks": ranks,
            }
            for replica, ranks in ((7, [0, 1]), (8, [2, 3]))
        ]
    )
    trainer._speco_start_rollout_idle_event_loop = lambda: (None, object())
    trainer._speco_bubble_session().start()
    assert {
        worker: (state.replica_rank, state.status)
        for worker, state in scheduler._idle_workers.items()
    } == {
        "0": (7, "generating"),
        "1": (7, "generating"),
        "2": (8, "generating"),
        "3": (8, "generating"),
    }


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_only_separate_pool_enables_strict_resource_scope(monkeypatch, mode):
    from verl_speco.integration import rollout_idle_events
    from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

    trainer = trainer_for(mode)
    trainer._drafter_scheduler.register_idle_training_resource_metadata(
        [
            {
                "in_drafter_train_group": True,
                "replica_rank": 7,
                "training_group_ranks": [0, 1],
                "full_collective_ranks": [0, 1],
            },
        ]
    )
    monkeypatch.setattr(
        SpecoRayPPOTrainer,
        "_speco_register_drafter_training_resource_metadata",
        lambda self: {},
    )
    trainer._speco_rollout_idle_event_bus_name = lambda: "standalone-bus"
    calls = []
    monkeypatch.setattr(
        rollout_idle_events,
        "configure_rollout_resource_groups",
        lambda *args, **kwargs: calls.append((args, kwargs)) or True,
    )
    trainer._speco_register_drafter_training_resource_metadata()
    assert calls == (
        [(("standalone-bus", {7: ("0", "1")}), {"strict_groups": True})]
        if mode == "separate_async"
        else []
    )


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("bad_ack", [False, True])
def test_head_poll_counts_transfer_only_after_complete_validated_ack(
    monkeypatch, mode, bad_ack
):
    from verl_speco.trainer import speco_ray_trainer as legacy
    from verl_speco.trainer.v1 import bubble_mixin

    trainer = trainer_for(mode)
    clock = [0.0]
    time = SimpleNamespace(time=lambda: 1000 + clock[0], perf_counter=lambda: clock[0])
    monkeypatch.setattr(legacy, "time", time)
    monkeypatch.setattr(bubble_mixin, "time", time)
    ready = [True]
    trainer._speco_ray_refs_ready = lambda refs: ready[0]
    trainer._speco_start_target_lm_head_weight_sync = lambda plan: (
        {},
        {
            "stage": "fetch",
            "target_version": 7,
            "target_worker_ids": ("0", "1"),
            "fetch_refs": ["fetch"],
            "sync_started": 0.0,
            "fetch_started": 0.0,
            "selected_rows": 0,
            "source_vocab_size": 0,
        },
    )

    def resolve(refs):
        if refs == ["fetch"]:
            clock[0] = 2.0
            return [{"export_strategy": "test"}]
        assert refs == ["ack"]
        clock[0] = 10.0
        return [
            {
                "worker_id": worker,
                "accepted": True,
                "global_step": 6 if bad_ack and worker == "1" else 7,
            }
            for worker in ("0", "1")
        ]

    trainer._ray_get_if_needed = resolve
    trainer._speco_build_drafter_target_lm_head_sync_args = (
        lambda payload, global_step: (
            payload,
            global_step,
            None,
        )
    )
    trainer.speco_sync_target_lm_head_weight = lambda *args, **kwargs: ["ack"]
    trainer._speco_v1_cache_old_policy_head()
    costs = trainer._speco_bubble_session().costs
    assert costs.total == 2
    dispatch = trainer._pending_target_lm_head_sync
    assert dispatch["sync_started"] == 0
    ready[0] = False
    clock[0] = 7
    trainer._speco_poll_target_lm_head_prefetch()
    assert costs.total == 2
    ready[0] = True
    if bad_ack:
        with pytest.raises(RuntimeError, match="Mismatched"):
            trainer._speco_wait_target_lm_head_prefetch()
        assert not any(key.startswith("head_transfer:") for key in costs.events)
        assert 7 not in trainer._speco_ready_target_lm_head_versions
    else:
        trainer._speco_wait_target_lm_head_prefetch()
        assert costs.total == 10
        assert costs.events["head_transfer:7:0.0"] == 10
        trainer._speco_poll_target_lm_head_prefetch()
        trainer._speco_finish_target_lm_head_weight_sync(dispatch)
        assert costs.total == 10
        assert trainer._speco_ready_target_lm_head_workers[7] == ("0", "1")
        clock[0] = 25
        trainer._speco_record_target_head_transfer_cost(
            {"target_version": 7, "sync_started": 20.0}
        )
        assert costs.total == 15


@pytest.mark.parametrize(
    "failure", [None, "drafter", "actor", "features", "manifest", "commit"]
)
@pytest.mark.parametrize("use_critic", [False, True])
@pytest.mark.parametrize("remove_previous", [False, True])
def test_native_joint_checkpoint_commits_last_and_preserves_previous_on_failure(
    tmp_path, monkeypatch, failure, use_critic, remove_previous
):
    from verl_speco.trainer.v1 import checkpoint_commit

    trainer = trainer_for("sync")
    trainer.config.trainer.update(
        {
            "default_local_dir": str(tmp_path),
            "default_hdfs_dir": None,
            "remove_previous_ckpt_in_save": remove_previous,
            "max_actor_ckpt_to_keep": 1,
            "max_critic_ckpt_to_keep": 2,
        }
    )
    trainer.use_critic = use_critic
    trainer.config["critic"] = {"checkpoint": {"async_save": False}}
    original = OmegaConf.to_container(trainer.config, resolve=True)
    trainer._speco_colocate_admission_closed = True
    trainer._speco_bubble_session().close = lambda boundary: {}
    trainer._speco_checkpoint_pending_quota = lambda: False
    trainer._speco_freeze_policy = None
    trainer._speco_freeze_save_state = lambda: None
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_freeze_branch_config = lambda: {}
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"step": 7})
    previous = tmp_path / "global_step_3"
    (previous / "actor").mkdir(parents=True)
    (previous / "actor" / "complete").write_text("previous")
    from verl.trainer.ppo.utils import Role

    critic_role = str(Role.Critic)
    (previous / critic_role).mkdir()
    (previous / critic_role / "complete").write_text("previous critic")
    (previous / "speco_v1_manifest.json").write_text(
        json.dumps(
            {
                "format": "speco_v1_joint_checkpoint",
                "version": 1,
                "global_step": 3,
                "bubble": {
                    "quota_complete": True,
                    "writer_group": [],
                    "published_plan_ids": [],
                    "policy_drafter_version": None,
                },
            }
        )
    )
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("3")
    failing = [failure]

    def check(stage):
        assert tracker.read_text() == "3"
        assert (previous / "actor" / "complete").exists()
        if failing[0] == stage:
            raise RuntimeError(stage)

    def actor_save(path, remote_path, step, max_ckpt_to_keep):
        check("actor")
        assert max_ckpt_to_keep is None
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "complete").write_text(str(step))

    worker = SimpleNamespace(
        execute_with_func_generator=SimpleNamespace(
            remote=lambda func, *args: func(None, *args),
        )
    )
    trainer.actor_rollout_wg = SimpleNamespace(
        save_checkpoint=actor_save, workers=[worker]
    )
    trainer.critic_wg = SimpleNamespace(save_checkpoint=actor_save, workers=[worker])
    trainer._ray_get_if_needed = lambda refs: refs
    trainer._speco_save_drafter_checkpoint = lambda **kwargs: check("drafter") or []
    trainer._speco_snapshot_v1_feature_store_checkpoint = (
        lambda: check("features") or []
    )
    write_manifest = trainer._speco_write_v1_joint_checkpoint_manifest

    def manifest(**kwargs):
        check("manifest")
        return write_manifest(**kwargs)

    trainer._speco_write_v1_joint_checkpoint_manifest = manifest
    commit = checkpoint_commit.commit_latest

    def checked_commit(root, step):
        check("commit")
        assert (tmp_path / "global_step_7" / "speco_v1_manifest.json").is_file()
        return commit(root, step)

    monkeypatch.setattr(checkpoint_commit, "commit_latest", checked_commit)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            trainer._save_checkpoint()
        assert tracker.read_text() == "3"
        assert (previous / "actor" / "complete").exists()
        assert getattr(trainer, "_speco_last_checkpoint_saved_step", None) is None
        assert OmegaConf.to_container(trainer.config, resolve=True) == original
        failing[0] = None
    trainer._save_checkpoint()
    assert tracker.read_text() == "7"
    assert (tmp_path / "global_step_7" / "actor" / "complete").is_file()
    assert not (previous / "actor").exists()
    assert (previous / critic_role / "complete").is_file() == (
        not use_critic or not remove_previous
    )
    assert trainer._speco_last_checkpoint_saved_step == 7
    assert OmegaConf.to_container(trainer.config, resolve=True) == original
