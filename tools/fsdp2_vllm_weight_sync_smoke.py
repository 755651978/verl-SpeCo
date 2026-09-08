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
"""Co-train FSDP2 actor -> external vLLM TP test; no checkpoint writes.

Run with torchrun, or use ``--ray`` to place every FSDP rank in a Ray actor.
See docs/fsdp2_vllm_weight_sync_smoke.md.
The actor is constructed by verl's production ``FSDPEngineWithLMHead``.  Its
real forward module and ``get_per_tensor_param()`` exporter are used directly.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import socket
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

# Prefer the checkout being tested over a possibly older pip-installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from verl.single_controller.base import Worker

from verl_speco.integration import external_vllm_weight_sync as weight_sync


_CONTROL_GROUP = None


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _RayRankActor(Worker):
    """One Ray actor matching one co-train actor/FSDP rank."""

    def __init__(
        self,
        rank,
        world_size,
        master_address,
        master_port,
        initialize_verl_worker_base,
    ):
        visible = [
            item
            for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
            if item
        ]
        local_rank = 0 if len(visible) == 1 else rank
        os.environ.update(
            {
                "MASTER_ADDR": str(master_address),
                "MASTER_PORT": str(master_port),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(local_rank),
            }
        )
        if initialize_verl_worker_base:
            super().__init__()
        self._smoke_rank = rank
        self._smoke_world_size = world_size
        self._smoke_local_rank = local_rank
        self._smoke_worker_base_initialized = bool(initialize_verl_worker_base)

    def run(self, argv):
        print(
            f"RAY_ACTOR_READY rank={self._smoke_rank} pid={os.getpid()} "
            f"local_rank={self._smoke_local_rank} "
            f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')!r} "
            f"verl_worker_base={int(self._smoke_worker_base_initialized)}",
            flush=True,
        )
        main(argv)
        return self._smoke_rank


def _run_with_ray(args, argv):
    import ray

    visible = [
        item
        for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if item
    ]
    if len(visible) < args.ray_num_workers:
        raise RuntimeError(
            f"--ray-num-workers={args.ray_num_workers} requires at least that many "
            f"ASCEND_RT_VISIBLE_DEVICES, got {visible}"
        )
    # This test owns a disposable local Ray runtime. It intentionally does not
    # connect to RAY_ADDRESS left over from a training job.
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(
        num_cpus=max(args.ray_num_workers * 2, 4),
        resources={"NPU": float(args.ray_num_workers)},
        include_dashboard=False,
    )
    actor_cls = ray.remote(num_cpus=1, resources={"NPU": 1})(_RayRankActor)
    worker_argv = [item for item in argv if item != "--ray"]
    master_port = _free_port()
    master_address = args.master_address or "127.0.0.1"
    actors = [
        actor_cls.remote(
            rank,
            args.ray_num_workers,
            master_address,
            master_port,
            args.initialize_verl_worker_base,
        )
        for rank in range(args.ray_num_workers)
    ]
    try:
        ray.get(
            [
                actor.run.remote(worker_argv) for actor in actors
            ]
        )
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()


def stage(label, action, *, rank0_only=False):
    """Propagate failures over CPU/Gloo, leaving HCCL free for weight setup."""
    error = None
    try:
        if not rank0_only or dist.get_rank() == 0:
            action()
    except Exception as exc:
        logging.exception("%s failed on rank %s", label, dist.get_rank())
        error = f"rank={dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error, group=_CONTROL_GROUP)
    if any(errors):
        raise RuntimeError(f"{label}: " + "; ".join(e for e in errors if e))
    if dist.get_rank() == 0:
        print(f"{label}_OK", flush=True)


@torch.no_grad()
def reference_logprobs(engine, token_batches):
    """Run the same wrapped module and BF16 autocast used by actor forward."""
    result = []
    with engine.eval_mode():
        for ids in token_batches:
            position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
                logits = engine.module(
                    input_ids=ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits[0, -1].float()
            result.append(torch.log_softmax(logits, dim=-1).cpu())
    return result


def build_cotrain_actor_engine(args):
    """Construct the actor through the same registry/config path as co-train."""
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import (
        FSDPEngineConfig,
        FSDPOptimizerConfig,
        HFModelConfig,
    )
    from verl.workers.engine import EngineRegistry

    model_config = HFModelConfig(
        path=args.model,
        trust_remote_code=False,
        use_remove_padding=True,
        enable_gradient_checkpointing=False,
    )
    model_config.model_type = "language_model"
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        forward_only=False,
        model_dtype="fp32",
        dtype="bfloat16",
        use_remove_padding=True,
        ulysses_sequence_parallel_size=1,
        param_offload=False,
        optimizer_offload=False,
    )
    optimizer_config = FSDPOptimizerConfig(
        lr=args.lr,
        lr_warmup_steps=0,
        total_training_steps=max(args.train_steps, 1),
    )
    engine = EngineRegistry.new(
        model_type="language_model",
        backend="fsdp2",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    return engine


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
            token_comparisons = []
            for encoded_id, actual_logprob in top.items():
                if not encoded_id.startswith("token_id:"):
                    raise RuntimeError(f"Expected token_id:<id>, got {encoded_id!r}")
                token_id = int(encoded_id.split(":", 1)[1])
                expected_logprob = float(expected[token_id])
                actual_logprob = float(actual_logprob)
                error = abs(expected_logprob - actual_logprob)
                if not math.isfinite(error):
                    raise RuntimeError("Non-finite token logprob comparison")
                errors.append(error)
                token_comparisons.append(
                    (token_id, expected_logprob, actual_logprob, error)
                )
                if old_references is not None:
                    largest_change = max(
                        largest_change,
                        abs(
                            float(expected[token_id] - old_references[index][token_id])
                        ),
                    )
            for rank, (
                token_id,
                expected_logprob,
                actual_logprob,
                error,
            ) in enumerate(token_comparisons, start=1):
                print(
                    f"TOP5_DETAIL prompt={index} rank={rank} token_id={token_id} "
                    f"hf_logprob={expected_logprob:.9f} "
                    f"vllm_logprob={actual_logprob:.9f} "
                    f"abs_error={error:.9f}",
                    flush=True,
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


def main(argv=None):
    global _CONTROL_GROUP

    argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Run each FSDP rank inside a Ray actor, as co-train does",
    )
    parser.add_argument("--ray-num-workers", type=int, default=2)
    parser.add_argument(
        "--process-group-layout",
        choices=("split", "verl-composite"),
        default="split",
        help=(
            "Use a pure accelerator default group plus a separate Gloo group, "
            "or verl's cpu:gloo,device:accelerator composite default group"
        ),
    )
    parser.add_argument(
        "--process-group-init",
        choices=("torch", "verl-ray"),
        default="torch",
        help="Initialize the default group directly or through verl's Ray helper",
    )
    parser.add_argument(
        "--install-verl-npu-vllm-compat",
        action="store_true",
        help="Apply the same process-wide NPU vLLM import patch as WorkerDict",
    )
    parser.add_argument(
        "--initialize-verl-worker-base",
        action="store_true",
        help="Run verl.single_controller.base.Worker.__init__ in every Ray actor",
    )
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Stop after the external vLLM communication group is initialized",
    )
    parser.add_argument(
        "--random-weight-test",
        action="store_true",
        help=(
            "First compare the normally-loaded vLLM with the same HF checkpoint, "
            "then randomize HF/FSDP2 weights, transfer them, and compare again"
        ),
    )
    parser.add_argument(
        "--random-weight-scale",
        type=float,
        default=0.01,
        help="Uniform random weights are sampled from [-scale, scale]",
    )
    args = parser.parse_args(argv)
    if (
        args.train_steps < 1
        or args.lr <= 0
        or args.logprob_atol <= 0
        or args.timeout <= 0
        or args.random_weight_scale <= 0
    ):
        parser.error("train-steps, lr, logprob-atol and timeout must be positive")
    if args.ray_num_workers < 1:
        parser.error("ray-num-workers must be positive")
    if args.ray:
        _run_with_ray(args, argv)
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device_module = torch.cuda
        device_type = "cuda"
        dist_backend = "nccl"
        runtime_version = f"CUDA={torch.version.cuda}"
    else:
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "This test requires CUDA or Ascend torch_npu"
            ) from exc
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable after importing torch_npu")
        device_module = npu
        device_type = "npu"
        dist_backend = "hccl"
        runtime_version = f"NPU={torch_npu.__version__}"
    device_module.set_device(local_rank)
    device = torch.device(device_type, local_rank)
    if args.install_verl_npu_vllm_compat:
        from verl_speco.integration.verl_npu_vllm_compat import (
            install_verl_npu_vllm_import_compat,
        )

        applied = install_verl_npu_vllm_import_compat()
        print(
            f"VERL_NPU_VLLM_COMPAT requested=1 applied={int(bool(applied))}",
            flush=True,
        )
    process_group_backend = dist_backend
    if args.process_group_layout == "verl-composite":
        process_group_backend = f"cpu:gloo,{device_type}:{dist_backend}"
    if args.process_group_init == "verl-ray":
        from verl.utils.distributed import initialize_global_process_group_ray

        initialize_global_process_group_ray(timeout_second=None)
        process_group_backend = "verl.initialize_global_process_group_ray"
    else:
        dist.init_process_group(
            process_group_backend, timeout=timedelta(seconds=args.timeout + 120)
        )
    # CONNECT is rank-asymmetric: rank 0 initializes a second NCCL/HCCL group
    # with external vLLM while peer actor ranks wait.  Using the actor's default
    # device group for that wait can overlap collectives in a different order.
    _CONTROL_GROUP = dist.new_group(
        backend="gloo",
        timeout=timedelta(seconds=args.timeout + 120),
    )
    if dist.get_rank() == 0:
        print("CONTROL_GROUP_READY backend=gloo", flush=True)
    worker = None
    try:
        logging.basicConfig(level=logging.WARNING)
        torch.manual_seed(1234)
        if dist.get_rank() == 0:
            print(f"SOURCE={weight_sync.__file__}", flush=True)
            print(
                f"TORCH={torch.__version__} {runtime_version} "
                f"DEVICE={device_type} BACKEND={process_group_backend} "
                f"PROCESS_GROUP_INIT={args.process_group_init} "
                f"PROCESS_GROUP_LAYOUT={args.process_group_layout} "
                "INSTALL_VERL_NPU_VLLM_COMPAT="
                f"{int(args.install_verl_npu_vllm_compat)} "
                "INITIALIZE_VERL_WORKER_BASE="
                f"{int(args.initialize_verl_worker_base)} "
                f"WORLD_SIZE={dist.get_world_size()}",
                flush=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        engine = build_cotrain_actor_engine(args)
        sample = next(engine.module.parameters())
        if not isinstance(sample, DTensor):
            raise RuntimeError("Co-train actor parameters are not FSDP2 DTensors")
        print(
            f"COTRAIN_ACTOR_READY rank={dist.get_rank()} "
            f"engine={type(engine).__module__}.{type(engine).__name__} "
            f"global_shape={tuple(sample.shape)} "
            f"local_shape={tuple(sample.to_local().shape)}",
            flush=True,
        )
        worker = SimpleNamespace(
            rank=dist.get_rank(),
            config=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2")),
            actor=SimpleNamespace(engine=engine),
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
        if args.connect_only:
            if dist.get_rank() == 0:
                print(
                    "RAY_COTRAIN_CONNECT_ONLY_PASS: external vLLM communication "
                    "group initialized",
                    flush=True,
                )
            return
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
        before = reference_logprobs(engine, token_batches)
        if args.random_weight_test:
            stage(
                "PHASE1_DISK_LOAD_COMPARE",
                lambda: check_vllm(args, token_batches, before),
                rank0_only=True,
            )
            stage(
                "PHASE2_ORIGINAL_WEIGHT_SYNC",
                lambda: weight_sync.update_worker_weights(worker, 0),
            )
            stage(
                "PHASE2_ORIGINAL_WEIGHT_COMPARE",
                lambda: check_vllm(args, token_batches, before),
                rank0_only=True,
            )
            torch.manual_seed(1234 + dist.get_rank())
            with torch.no_grad():
                for parameter in engine.module.parameters():
                    local_parameter = (
                        parameter.to_local()
                        if isinstance(parameter, DTensor)
                        else parameter
                    )
                    local_parameter.uniform_(
                        -args.random_weight_scale,
                        args.random_weight_scale,
                    )
            random_references = reference_logprobs(engine, token_batches)
            stage(
                "PHASE3_RANDOM_WEIGHT_SYNC",
                lambda: weight_sync.update_worker_weights(worker, 1),
            )
            stage(
                "PHASE3_RANDOM_WEIGHT_COMPARE",
                lambda: check_vllm(
                    args,
                    token_batches,
                    random_references,
                    old_references=before,
                ),
                rank0_only=True,
            )
            if dist.get_rank() == 0:
                print(
                    "PASS: disk-loaded, transferred original, and transferred "
                    "random weights all passed comparison",
                    flush=True,
                )
            return
        stage("INITIAL_SYNC", lambda: weight_sync.update_worker_weights(worker, 0))
        stage(
            "INITIAL_COMPARE",
            lambda: check_vllm(args, token_batches, before),
            rank0_only=True,
        )
        # A disposable update through the optimizer built by the co-train engine.
        training_ids = tokenizer(
            "The capital of France is Paris. One plus one equals two. "
            "A neural network learns patterns from examples.",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        optimizer = engine.optimizer
        engine.module.train()
        for step in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = engine.module(
                input_ids=training_ids, labels=training_ids, use_cache=False
            ).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training produced non-finite loss")
            loss.backward()
            engine.optimizer_step()
            if dist.get_rank() == 0:
                print(
                    f"TRAIN_STEP_OK step={step + 1} loss={float(loss.detach()):.6f}",
                    flush=True,
                )
        optimizer.zero_grad(set_to_none=True)
        after = reference_logprobs(engine, token_batches)
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
                "PASS: initial and trained FSDP2 weights match external vLLM; "
                "no checkpoint written",
                flush=True,
            )
    finally:
        try:
            if worker is not None:
                weight_sync.close_worker_weight_sync(worker)
        finally:
            # Do not insert a barrier in error cleanup. torchrun handles failed peers.
            if _CONTROL_GROUP is not None:
                dist.destroy_process_group(_CONTROL_GROUP)
                _CONTROL_GROUP = None
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
