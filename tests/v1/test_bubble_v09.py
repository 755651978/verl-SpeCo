# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Exercise the real upstream 0.9 PPO loop and mode hooks on CPU fakes."""

import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_speco.trainer.v1.factory import _trainer_types

try:
    TRAINERS = _trainer_types()
except (ImportError, RuntimeError):
    if os.getenv("SPECO_REQUIRE_V1") == "1":
        raise
    pytest.skip("requires upstream verl 0.9 and TransferQueue", allow_module_level=True)


def config(mode):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "trainer": {
                "device": "cpu",
                "critic_warmup": 0,
                "v1": {"trainer_mode": mode},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 1.0,
                    "name": "vllm",
                    "drafter": {
                        "enable": True,
                        "enable_drafter_training": True,
                        "training": {
                            "collect_hidden_states_from_old_logprob": True,
                            "scheduler": {
                                "execution": {"strategy": "rollout_idle_worker"}
                            },
                        },
                    },
                }
            },
        }
    )


@pytest.mark.parametrize("mode", list(("sync", "colocate_async", "separate_async")))
def test_real_native_step_obeys_bubble_resource_boundary(mode):
    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = 1
    trainer.parameter_sync_step = 2
    events = []
    session = SimpleNamespace(
        start=lambda: events.append("bubble_start"),
        close=lambda boundary: events.append("bubble_close") or {},
    )
    trainer._speco_bubble_lifecycle = session
    trainer._speco_v1_admission = lambda method: events.append(method)
    trainer._add_batch_to_generate = lambda: events.append("enqueue")
    trainer.checkpoint_manager = SimpleNamespace(
        abort_replicas=lambda: events.append("abort"),
        sleep_replicas=lambda: events.append("sleep"),
    )
    if mode == "separate_async":
        from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode

        trainer.current_mode = HybridEngineMode.ROLLOUT
        trainer.switch_to_trainer = lambda: events.append("hybrid_to_trainer")
    batches = iter(
        SimpleNamespace(
            extra_info={},
            tags=[{"is_padding": False}],
            keys=[f"row{index}"],
            partition_id="train",
        )
        for index in range(2)
    )
    trainer.replay_buffer = SimpleNamespace(sample=lambda **kwargs: (next(batches), {}))
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=[object()])
    trainer.use_reference_policy = trainer.use_critic = False
    trainer._balance_batch = lambda value, **kwargs: value
    trainer._compute_advantage = lambda value, **kwargs: value

    def old_log_prob(value, **kwargs):
        events.append("old_log_prob")
        assert ("bubble_close" in events) == (mode != "separate_async")
        return value

    trainer._compute_old_log_prob = old_log_prob
    trainer._update_actor = lambda value, **kwargs: events.append("actor") or value
    result = trainer.step({}, {})
    assert result.keys == ["row0", "row1"]
    assert events.index("bubble_start") < events.index("enqueue")
    assert events.count("old_log_prob") == events.count("actor") == 2
    if mode != "separate_async":
        assert (
            events.index("bubble_close")
            < events.index("sleep")
            < events.index("old_log_prob")
        )


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_mode_step_end_holds_target_across_probe_pair(mode, checkpoint):
    from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle

    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = 1
    trainer.total_training_steps = 5
    trainer.timing_raw = {}
    events = []
    target = [0]

    def target_sync(*args):
        target[0] += 1
        events.append("target_sync")

    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=target_sync,
        abort_replicas=lambda: events.append("abort"),
        resume_generation_replicas=lambda: events.append("native_resume"),
        sleep_replicas=lambda: events.append("sleep"),
    )
    trainer.standalone_checkpoint_manager = trainer.checkpoint_manager
    manager = SimpleNamespace(server_handles=[])
    trainer.llm_server_manager = trainer.standalone_server_manager = manager
    if mode == "separate_async":
        from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode

        trainer.current_mode = HybridEngineMode.TRAINER
    trainer._speco_v1_admission = lambda method: events.append(method)
    trainer._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        _training_quota_debt_steps=0,
        training_work_metrics=lambda: {},
    )
    trainer._speco_wait_target_lm_head_prefetch = lambda: None
    trainer._speco_freeze_policy = SimpleNamespace(
        drafter_version=0, probe_due=lambda version: True
    )
    trainer._speco_freeze_metrics = lambda: {}
    session = BubbleLifecycle(trainer)
    session.close = lambda boundary: {}
    session.start = lambda: events.append("bubble_resume")
    trainer._speco_bubble_lifecycle = session
    plan = SimpleNamespace(plan_id="quota", publish_after_success=True)
    session.enqueue(plan, SimpleNamespace(trained=True, raw_results=[], elapsed_sec=4))
    probes = []

    def probe():
        probes.append(target[0])
        events.append("probe")
        return [], 0

    trainer._speco_run_freeze_probe_arm = probe
    trainer._speco_build_freeze_probe_comparison = lambda *args, **kwargs: "paired"

    def publish(*args, **kwargs):
        events.append("publish")
        trainer._speco_v1_publish_ack = True
        return {"drafter/published": 1}

    trainer._speco_publish_drafter_weights = publish
    trainer._speco_wait_pending_drafter_publish = lambda: None
    observations = []
    trainer._speco_freeze_observe_publish = (
        lambda metrics, **kwargs: observations.append(kwargs)
    )
    if checkpoint:
        trainer._speco_deferred_checkpoint_step = 1

        def save():
            assert len(observations) == 1
            assert not session.pending
            events.append("checkpoint")

        trainer._save_checkpoint = save
    trainer.on_step_end()
    assert probes == [1, 1]
    assert events.index("speco_close_rollout_admission") < events.index("target_sync")
    assert events.index("publish") < events.index("speco_resume_rollout_admission")
    assert len(observations) == 1
    assert observations[0]["probe"] == "paired"
    assert observations[0]["update_cost_seconds"] >= 4
    assert not session.pending
    if checkpoint:
        assert events.index("publish") < events.index("checkpoint")


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_real_mode_final_quota_cannot_enable_critical_path_fallback(mode):
    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = trainer.total_training_steps = 1
    trainer._speco_bubble_lifecycle = SimpleNamespace(
        close=lambda boundary: {}, pending=[]
    )
    trainer._speco_get_drafter_scheduler = lambda: SimpleNamespace(
        _training_quota_debt_steps=1
    )
    trainer._speco_wait_target_lm_head_prefetch = lambda: None
    trainer._speco_v1_admission = lambda method: None
    trainer._speco_drafter_schedule_config = lambda: SimpleNamespace(
        training_quota_allow_critical_path_fallback=False,
    )
    trainer.checkpoint_manager = trainer.standalone_checkpoint_manager = (
        SimpleNamespace(
            abort_replicas=lambda: None,
        )
    )
    trainer._speco_bubble_lifecycle.costs = SimpleNamespace(
        record_interval=lambda *args: None
    )
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        trainer.on_step_end()
    assert trainer._speco_colocate_admission_closed


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
@pytest.mark.parametrize("final", [False, True])
def test_explicit_topup_preserves_bounded_and_final_completion_policy(mode, final):
    from verl_speco.trainer.bubble_lifecycle import BubbleLifecycle

    trainer = TRAINERS[mode].__new__(TRAINERS[mode])
    trainer.config = config(mode)
    trainer.global_steps = 1
    trainer.total_training_steps = 1 if final else 5
    trainer.timing_raw = {}
    trainer._speco_v1_admission = lambda method: None
    trainer._speco_wait_target_lm_head_prefetch = lambda: None
    trainer._speco_park_drafter_training_memory = lambda boundary: {}
    trainer._speco_freeze_policy = None
    manager = SimpleNamespace(
        abort_replicas=lambda: None,
        sleep_replicas=lambda: None,
        update_weights=lambda *args: None,
        resume_generation_replicas=lambda: None,
    )
    trainer.checkpoint_manager = trainer.standalone_checkpoint_manager = manager
    scheduler = SimpleNamespace(
        _training_quota_debt_steps=3, training_work_metrics=lambda: {}
    )
    trainer._speco_get_drafter_scheduler = lambda: scheduler
    trainer._speco_drafter_schedule_config = lambda: SimpleNamespace(
        training_quota_allow_critical_path_fallback=True,
    )
    trainer._speco_drafter_schedule_context = lambda: object()
    scheduler._plan_training_quota_topup = lambda *args: SimpleNamespace(
        plan_id="regular-topup",
        launch=True,
        publish_after_success=True,
    )
    scheduler.prepare_training_execution = lambda plan: {}
    session = BubbleLifecycle(trainer)
    session.close = lambda boundary: {}
    session.start = lambda: None
    trainer._speco_bubble_lifecycle = session
    calls = []

    def schedule(**kwargs):
        assert kwargs["force_quota_completion"]
        return SimpleNamespace(
            training_plan=SimpleNamespace(
                plan_id=f"plan-{scheduler._training_quota_debt_steps}",
                launch=True,
                publish_after_success=True,
            )
        )

    def train(plan):
        calls.append(plan.plan_id)
        scheduler._training_quota_debt_steps -= 1
        return True, {}

    trainer._speco_on_before_actor_update = schedule
    trainer._speco_train_drafter = train
    publications = []
    trainer._speco_publish_drafter_weights = lambda trained, plan, **kwargs: (
        publications.append(plan.plan_id) or {"drafter/published": 1}
    )
    trainer._speco_wait_pending_drafter_publish = lambda: None
    trainer._speco_freeze_observe_publish = lambda *args, **kwargs: None
    trainer.on_step_end()
    assert calls == (["plan-3", "plan-2", "plan-1"] if final else ["regular-topup"])
    assert publications == (["plan-1"] if final else [])
    assert scheduler._training_quota_debt_steps == (0 if final else 2)
