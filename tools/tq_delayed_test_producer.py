# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Publish protocol-valid synthetic DSpark features in delayed batches.

This is an integration-test Producer for the standalone TQ Consumer.  It does
not run vLLM: tensor shapes are derived from the target model config so the
normal DSpark preprocessing and training path can consume the samples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from verl_speco.integration.transferqueue_bridge import (
    close_transfer_queue_client,
    configure_transfer_queue,
    connect_ray_cluster,
    connect_transfer_queue_client,
    list_samples,
    put_sample,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.transport.drafter_sample_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    SampleMetadata,
    encode_sample,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
)


def _tq_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enable": True,
        "package_version": "0.1.10",
        "ray": {"address": args.ray_address, "namespace": args.namespace},
        "partition_id": args.partition_id,
        "run_id": args.run_id,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 100000,
                "num_data_storage_units": 8,
            },
        },
    }


def _model_dimensions(model_path: str) -> tuple[int, int, int]:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config") or config
    hidden_size = int(text_config["hidden_size"])
    vocab_size = int(text_config["vocab_size"])
    num_hidden_layers = int(text_config.get("num_hidden_layers", 1))
    return hidden_size, vocab_size, num_hidden_layers


def _wait_for_owner(args: argparse.Namespace) -> None:
    owner_ready_key = f"control:v{PROTOCOL_SCHEMA_VERSION}:{args.run_id}:owner-ready"
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        tag = list_samples().get(owner_ready_key)
        if isinstance(tag, dict) and tag.get("status") == "owner_ready":
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for TQ owner key {owner_ready_key!r}")


def _target_layer_ids(num_target_layers: int, num_hidden_layers: int) -> list[int]:
    if num_target_layers <= 0:
        raise ValueError("num_target_layers must be positive")
    if num_hidden_layers < 4:
        raise ValueError("the DSpark target model must have at least four hidden layers")
    if num_target_layers == 1:
        return [num_hidden_layers // 2]
    start = 1
    end = num_hidden_layers - 3
    span = end - start
    return [
        int(round(start + (index * span) / (num_target_layers - 1)))
        for index in range(num_target_layers)
    ]


def _sample(
    args: argparse.Namespace,
    *,
    sequence_no: int,
    hidden_size: int,
    vocab_size: int,
    target_layer_ids: list[int],
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + sequence_no)
    # The default standalone DSpark configuration enables L1 distillation.
    # Its wire layout contains N auxiliary target layers followed by the
    # target model's final hidden state, all concatenated on the last axis.
    hidden_dim = hidden_size * (len(target_layer_ids) + 1)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.sequence_length,),
        generator=generator,
        dtype=torch.long,
    )
    loss_mask = torch.ones(args.sequence_length, dtype=torch.float32)
    loss_mask[: max(1, args.sequence_length // 4)] = 0
    hidden_states = torch.randn(
        args.sequence_length,
        hidden_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    sample_id = f"consumer-test-{sequence_no:08d}"
    metadata = SampleMetadata(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        run_id=args.run_id,
        sample_id=sample_id,
        sequence_no=sequence_no,
    )
    sample = DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=input_ids,
        loss_mask=loss_mask,
        position_ids=torch.arange(args.sequence_length, dtype=torch.long),
        hidden_states=hidden_states,
        metadata={"hidden_states_layout": "dflash_aux_plus_last"},
    )
    key = make_sample_key(metadata)
    return key, encode_sample(sample, metadata), make_ready_tag(metadata)


def run(args: argparse.Namespace) -> None:
    hidden_size, vocab_size, num_hidden_layers = _model_dimensions(args.model_path)
    target_layer_ids = _target_layer_ids(args.num_target_layers, num_hidden_layers)
    samples_per_batch = args.world_size * args.batch_size_per_gpu
    total_samples = args.num_batches * samples_per_batch
    config = _tq_config(args)
    configure_transfer_queue(config)
    connect_ray_cluster(args.ray_address, args.namespace)
    connect_transfer_queue_client()
    try:
        _wait_for_owner(args)
        print(
            "PRODUCER_CONNECTED "
            f"batches={args.num_batches} samples_per_batch={samples_per_batch} "
            f"sequence_length={args.sequence_length} hidden_shape="
            f"({args.sequence_length}, {hidden_size * (args.num_target_layers + 1)})",
            flush=True,
        )
        time.sleep(args.initial_delay)
        sequence_no = 0
        for batch_index in range(args.num_batches):
            keys = []
            for _ in range(samples_per_batch):
                key, fields, tag = _sample(
                    args,
                    sequence_no=sequence_no,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    target_layer_ids=target_layer_ids,
                )
                put_sample(key, fields, tag=tag)
                keys.append(key)
                sequence_no += 1
            print(
                f"PRODUCER_BATCH_READY batch={batch_index + 1}/{args.num_batches} "
                f"samples={len(keys)} sequence_no=[{sequence_no - len(keys)},{sequence_no})",
                flush=True,
            )
            if batch_index + 1 < args.num_batches:
                time.sleep(args.batch_interval)
        eos_key, eos_fields, eos_tag = make_eos_record(args.run_id, total_samples)
        put_sample(eos_key, eos_fields, tag=eos_tag)
        print(f"PRODUCER_EOS total_samples={total_samples} key={eos_key}", flush=True)
    finally:
        close_transfer_queue_client()
        print("PRODUCER_CLOSED_LOCAL", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--namespace", default="speco-drafter")
    parser.add_argument("--partition-id", default="speco_drafter_features")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--num-target-layers", type=int, default=5)
    parser.add_argument("--initial-delay", type=float, default=5.0)
    parser.add_argument("--batch-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.world_size <= 0 or args.batch_size_per_gpu <= 0:
        parser.error("world-size and batch-size-per-gpu must be positive")
    if args.num_batches <= 0 or args.sequence_length <= 0:
        parser.error("num-batches and sequence-length must be positive")
    run(args)


if __name__ == "__main__":
    main()
