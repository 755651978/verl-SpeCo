# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import asyncio
from types import MethodType

import pytest


def test_accumulation_uses_combined_valid_token_mean(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="drafter accumulation needs the trainer dependency stack",
    )
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    class _Backend:
        model_type = "test"

        def __init__(self, model):
            self.model = model

        def compute_loss(self, _model, batch, _pad_size):
            tokens = batch["tokens"].float()
            total = self.model.weight * batch["loss_sum_factor"].float()
            return {
                "total_local_vloss": total * 0.0,
                "total_local_ploss": total,
                "local_num_tokens": tokens,
                "v_weight": 0.0,
                "p_weight": 1.0,
            }

    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.model = _Model()
    trainer.backend = _Backend(trainer.model)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.lr_scheduler = None
    trainer.use_ulysses_sp = False
    trainer.training_steps = 0
    trainer.optimizer_steps_total = 0
    trainer._current_pad_size = 0
    trainer._current_accumulation_valid_tokens = 0
    trainer._current_accumulation_vloss_sum = 0.0
    trainer._current_accumulation_ploss_sum = 0.0
    trainer._last_optimizer_valid_tokens = 0
    trainer.record_training_timing = lambda *_args, **_kwargs: None
    trainer._record_dflash_training_metrics = lambda *_args, **_kwargs: None
    trainer._get_sp_group = lambda: None
    trainer._reduce_loss_metrics = MethodType(
        lambda _self, l_v, l_p, l_n: (l_v, l_p, l_n, 1), trainer
    )

    async def _run() -> None:
        assert await trainer._training_step_on_batch(
            {
                "tokens": torch.tensor(1.0),
                "loss_sum_factor": torch.tensor(1.0),
            },
            1,
            accumulation_steps=2,
            accumulation_index=0,
        )
        assert await trainer._training_step_on_batch(
            {
                "tokens": torch.tensor(3.0),
                "loss_sum_factor": torch.tensor(6.0),
            },
            1,
            accumulation_steps=2,
            accumulation_index=1,
        )

    asyncio.run(_run())

    # (1 + 6) / (1 + 3) = 1.75; averaging micro-batch means would be 1.5.
    assert trainer.model.weight.item() == pytest.approx(-1.75)
    assert trainer._last_optimizer_valid_tokens == 4
