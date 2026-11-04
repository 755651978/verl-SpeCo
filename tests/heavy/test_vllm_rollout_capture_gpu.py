# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Opt-in CUDA numerical acceptance test against a same-weight HF target.

Set SPECO_CAPTURE_TARGET and SPECO_CAPTURE_DRAFT to compatible EAGLE3 model
paths. This test intentionally runs reference forwards; production capture does
not. Run in a fresh process because runner hooks and CUDA graphs are process-wide.
"""

import os

import pytest


@pytest.mark.skipif(
    not (os.getenv("SPECO_CAPTURE_TARGET") and os.getenv("SPECO_CAPTURE_DRAFT")),
    reason="requires CUDA and explicit compatible target/draft model paths",
)
@pytest.mark.parametrize("eager", [True, False])
def test_rollout_features_match_reference(eager, monkeypatch, tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verl_speco.integration import vllm_rollout_capture as capture

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(capture, "capture_root", lambda: tmp_path)
    capture.install_runner_capture()
    from vllm import LLM, SamplingParams

    target, draft = (
        os.environ["SPECO_CAPTURE_TARGET"],
        os.environ["SPECO_CAPTURE_DRAFT"],
    )
    tokenizer = AutoTokenizer.from_pretrained(target)
    prompt = tokenizer.encode("Explain why the sky appears blue in two sentences.")
    plan = capture.make_plan(
        {"hidden_state_window_tokens_per_sample": 32},
        prompt_len=len(prompt),
        max_tokens=32,
        request_id="gpu",
        global_step=1,
        replica_rank=0,
    )
    plan["layout"] = "eagle3_aux_plus_last"
    directory = capture.plan_directory(plan)
    directory.mkdir()
    (directory / "live").touch()
    engine = LLM(
        model=target,
        speculative_config={
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 3,
        },
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=eager,
        gpu_memory_utilization=0.5,
        max_model_len=512,
    )
    try:
        output = engine.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                temperature=0, max_tokens=32, extra_args={capture.CAPTURE_KEY: plan}
            ),
        )[0]
        response = list(output.outputs[0].token_ids)
        sample = capture.load_sample(plan, prompt, response)
        assert sample is not None
        # Keep the reference on CPU so the engine and HF weights do not compete
        # for GPU memory. Use the same target dtype for a useful comparison.
        reference = AutoModelForCausalLM.from_pretrained(
            target, torch_dtype=torch.bfloat16
        )
        with torch.inference_mode():
            states = reference(
                torch.tensor([prompt + response[:-1]]),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
        positions = sample["hidden_positions"][0]
        expected = torch.cat(
            [states[i][0, positions] for i in sample["hidden_layer_ids"]]
            + [states[-1][0, positions]],
            dim=-1,
        ).float()
        actual = sample["hidden_states"][0].float()
        width = states[-1].shape[-1]
        for a, b in zip(
            actual.split(width, dim=-1), expected.split(width, dim=-1), strict=True
        ):
            relative_rmse = (
                a - b
            ).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)
            assert relative_rmse.item() < 0.03
    finally:
        capture.cleanup(plan)
        del engine
