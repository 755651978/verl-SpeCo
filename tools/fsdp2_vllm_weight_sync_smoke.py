#!/usr/bin/env python3
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
"""Manual CUDA FSDP2 -> external vLLM TP smoke test; no Ray or checkpoint writes.

Run with torchrun. See docs/fsdp2_vllm_weight_sync_smoke.md.
The adapter below is test-only: it uses native FSDP2 for dense HF models, not
verl's engine-specific MoE/LoRA parameter conversions.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

# Prefer the checkout being tested over a possibly older pip-installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl_speco.integration import external_vllm_weight_sync as weight_sync


class FSDP2ExportEngine:
    """Provide the production sender's engine API using native FSDP2."""

    is_param_offload_enabled = False

    def __init__(self, model):
        self.model = model

    def get_per_tensor_param(self, **kwargs):
        def weights():
            for name, value in self.model.state_dict().items():
                # BOTH FSDP ranks must consume this collective iterator.
                yield name, value.full_tensor() if isinstance(value, DTensor) else value

        return weights(), None


def stage(label, action, *, rank0_only=False):
    """Propagate HTTP/Python failures before another rank starts the next stage."""
    error = None
    try:
        if not rank0_only or dist.get_rank() == 0:
            action()
    except Exception as exc:
        logging.exception("%s failed on rank %s", label, dist.get_rank())
        error = f"rank={dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"{label}: " + "; ".join(e for e in errors if e))
    if dist.get_rank() == 0:
        print(f"{label}_OK", flush=True)


@torch.no_grad()
def reference_logprobs(model, token_batches):
    model.eval()
    result = []
    for ids in token_batches:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        result.append(torch.log_softmax(logits, dim=-1).cpu())
    return result


def check_vllm(args, token_batches, references, *, old_references=None):
    """Compare top-k logprobs by token ID, never by decoded text."""
    largest_change = 0.0
    with requests.Session() as session:
        session.trust_env = False
        for index, (ids, expected) in enumerate(
            zip(token_batches, references, strict=True)
        ):
            response = session.post(
                args.endpoint.rstrip("/") + "/completions",
                json={
                    "model": args.served_model_name,
                    "prompt": ids[0].tolist(),
                    "max_tokens": 1,
                    "temperature": 0,
                    "repetition_penalty": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "ignore_eos": True,
                    "logprobs": 5,
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                    "seed": 0,
                },
                timeout=args.timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"completions HTTP {response.status_code}: {response.text[:2000]}"
                )
            choice = response.json()["choices"][0]
            actual_id = int(choice["token_ids"][0])
            top = choice["logprobs"]["top_logprobs"][0]
            if not top:
                raise RuntimeError("vLLM returned no top_logprobs")
            errors = []
            for encoded_id, actual_logprob in top.items():
                if not encoded_id.startswith("token_id:"):
                    raise RuntimeError(f"Expected token_id:<id>, got {encoded_id!r}")
                token_id = int(encoded_id.split(":", 1)[1])
                error = abs(float(expected[token_id]) - float(actual_logprob))
                if not math.isfinite(error):
                    raise RuntimeError("Non-finite token logprob comparison")
                errors.append(error)
                if old_references is not None:
                    largest_change = max(
                        largest_change,
                        abs(
                            float(expected[token_id] - old_references[index][token_id])
                        ),
                    )
            gap = float(expected.max() - expected[actual_id])
            max_error = max(errors)
            print(
                f"COMPARE prompt={index} hf_top1={int(expected.argmax())} "
                f"vllm_top1={actual_id} top1_gap={gap:.6f} "
                f"max_logprob_error={max_error:.6f} tolerance={args.logprob_atol}",
                flush=True,
            )
            if not torch.isfinite(torch.tensor([gap, max_error])).all():
                raise RuntimeError("Non-finite comparison values")
            if gap > args.logprob_atol or max_error > args.logprob_atol:
                raise RuntimeError(
                    "HF/FSDP2 and vLLM outputs do not match within tolerance"
                )
    if old_references is not None:
        print(f"CHECKED_TOKEN_REFERENCE_CHANGE={largest_change:.6f}", flush=True)
        if largest_change <= 2 * args.logprob_atol:
            raise RuntimeError(
                "INCONCLUSIVE: training changed the checked logits too little to "
                "rule out stale weights at this tolerance. Restart the smoke service "
                "and retry with --train-steps 3 --lr 0.05. Do not treat this as PASS."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Local dense Qwen/Llama HF model directory"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model-name", default="weight-sync-smoke")
    parser.add_argument(
        "--master-address",
        default=None,
        help="Actor rank 0 address reachable from vLLM",
    )
    parser.add_argument("--bucket-size-mb", type=int, default=256)
    parser.add_argument("--no-packed", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--logprob-atol", type=float, default=0.1)
    args = parser.parse_args()
    if (
        args.train_steps < 1
        or args.lr <= 0
        or args.logprob_atol <= 0
        or args.timeout <= 0
    ):
        parser.error("train-steps, lr, logprob-atol and timeout must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test requires CUDA; verify the driver/PyTorch versions first"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=args.timeout + 120), device_id=device
    )
    worker = None
    try:
        logging.basicConfig(level=logging.WARNING)
        torch.manual_seed(1234)
        if dist.get_world_size() < 2:
            raise ValueError(
                "Use torchrun --nproc_per_node=2 to test real FSDP sharding"
            )
        if dist.get_rank() == 0:
            print(f"SOURCE={weight_sync.__file__}", flush=True)
            print(f"TORCH={torch.__version__} CUDA={torch.version.cuda}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        if (
            int(getattr(model.config, "num_experts", 0) or 0) > 0
            or int(getattr(model.config, "num_local_experts", 0) or 0) > 0
        ):
            raise ValueError(
                "This smoke adapter covers dense models only; use a dense Qwen/Llama model"
            )
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Expected a dense HF model with model.model.layers")
        mesh = init_device_mesh("cuda", (dist.get_world_size(),))
        # Load on CPU, then shard/move layer by layer; do not .cuda() the whole model.
        for layer in layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        sample = next(model.parameters())
        if not isinstance(sample, DTensor):
            raise RuntimeError("Model parameters are not FSDP2 DTensors")
        print(
            f"FSDP_READY rank={dist.get_rank()} global_shape={tuple(sample.shape)} local_shape={tuple(sample.to_local().shape)}",
            flush=True,
        )
        worker = SimpleNamespace(
            rank=dist.get_rank(),
            config=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2")),
            actor=SimpleNamespace(engine=FSDP2ExportEngine(model)),
        )
        cfg = {
            "endpoints": [args.endpoint],
            "master_address": args.master_address,
            "bucket_size_mb": args.bucket_size_mb,
            "packed": not args.no_packed,
            "timeout_seconds": args.timeout,
            "packed_num_buffers": 2,
        }
        stage("CONNECT", lambda: weight_sync.initialize_worker_weight_sync(worker, cfg))
        prompts = [
            "The capital of France is",
            "One plus one equals",
            "Explain what a neural network is:",
        ]
        token_batches = [
            tokenizer(text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(device)
            for text in prompts
        ]
        before = reference_logprobs(model, token_batches)
        stage("INITIAL_SYNC", lambda: weight_sync.update_worker_weights(worker, 0))
        stage(
            "INITIAL_COMPARE",
            lambda: check_vllm(args, token_batches, before),
            rank0_only=True,
        )
        # A disposable real training step. SGD avoids Adam optimizer-state memory.
        training_ids = tokenizer(
            "The capital of France is Paris. One plus one equals two. "
            "A neural network learns patterns from examples.",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        model.train()
        for step in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=training_ids, labels=training_ids, use_cache=False
            ).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training produced non-finite loss")
            loss.backward()
            optimizer.step()
            if dist.get_rank() == 0:
                print(
                    f"TRAIN_STEP_OK step={step + 1} loss={float(loss.detach()):.6f}",
                    flush=True,
                )
        optimizer.zero_grad(set_to_none=True)
        after = reference_logprobs(model, token_batches)
        stage(
            "UPDATED_SYNC",
            lambda: weight_sync.update_worker_weights(worker, args.train_steps),
        )
        stage(
            "UPDATED_COMPARE",
            lambda: check_vllm(args, token_batches, after, old_references=before),
            rank0_only=True,
        )
        if dist.get_rank() == 0:
            print(
                "PASS: initial and trained FSDP2 weights match external vLLM; no checkpoint written",
                flush=True,
            )
    finally:
        try:
            if worker is not None:
                weight_sync.close_worker_weight_sync(worker)
        finally:
            # Do not insert a barrier in error cleanup. torchrun handles failed peers.
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
