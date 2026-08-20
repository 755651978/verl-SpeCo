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
"""Real two-process smoke test for Ray + TransferQueue 0.1.7.

Start a Ray head first, then run ``owner`` and ``client`` in separate shells.
The owner publishes one protocol-valid sample; the client list/get/decodes and
clears it, publishes a done marker, and exits without killing the owner.
"""

from __future__ import annotations

import argparse
import time

import torch

from verl_speco.integration.transferqueue_bridge import (
    close_transfer_queue_owner,
    configure_transfer_queue,
    connect_ray_cluster,
    list_samples,
    put_sample,
    start_transfer_queue_owner,
)
from verl_speco.trainer.feature_store import DraftFeatureSample
from verl_speco.trainer.tq_feature_store import TQFeatureStore
from verl_speco.trainer.tq_sample_source import TQFeatureDataLoader
from verl_speco.transport.drafter_sample_protocol import (
    SampleMetadata,
    encode_sample,
    make_eos_record,
    make_ready_tag,
    make_sample_key,
)


def _config(args) -> dict:
    return {
        "enable": True,
        "package_version": "0.1.7",
        "ray": {"address": args.ray_address, "namespace": args.namespace},
        "partition_id": "speco_drafter_features",
        "run_id": args.run_id,
        "schema_version": 1,
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 32,
                "num_data_storage_units": 1,
            },
        },
    }


def _record(run_id: str, sequence_no: int):
    meta = SampleMetadata(
        schema_version=1,
        run_id=run_id,
        sample_id=f"smoke-{sequence_no:04d}",
        sequence_no=sequence_no,
        algorithm="DSPARK",
        target_model_id="smoke-target",
        target_model_revision="smoke-revision",
        tokenizer_fingerprint="smoke-tokenizer",
        target_layer_ids=[0],
        hidden_states_layout="dflash_aux",
        hidden_dtype="float32",
        hidden_shape=[3, 4],
        feature_length=3,
        full_sequence_length=3,
        feature_start=0,
        feature_end=3,
        use_logits=False,
    )
    sample = DraftFeatureSample(
        algorithm="DSPARK",
        input_ids=torch.tensor([1, 2, 3]) + sequence_no,
        loss_mask=torch.tensor([0.0, 1.0, 1.0]),
        position_ids=torch.tensor([0, 1, 2]),
        hidden_states=(
            torch.arange(12, dtype=torch.float32).reshape(3, 4) + sequence_no
        ),
    )
    return meta, sample


def _wait_for(predicate, timeout: float, description: str):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {description}")


def run_owner(args) -> None:
    config = _config(args)
    configure_transfer_queue(config)
    connect_ray_cluster(args.ray_address, args.namespace)
    start_transfer_queue_owner(config)
    records = [_record(args.run_id, sequence_no) for sequence_no in range(2)]
    keys = [make_sample_key(meta) for meta, _ in records]
    try:
        owner_ready_key = f"control:v1:{args.run_id}:owner-ready"
        put_sample(
            owner_ready_key,
            {"marker": torch.tensor([1], dtype=torch.uint8)},
            tag={
                "record_type": "control",
                "status": "owner_ready",
                "schema_version": 1,
                "run_id": args.run_id,
            },
        )
        for (meta, sample), key in zip(records, keys, strict=True):
            put_sample(key, encode_sample(sample, meta), tag=make_ready_tag(meta))
        eos_key, eos_fields, eos_tag = make_eos_record(args.run_id, len(records))
        put_sample(eos_key, eos_fields, tag=eos_tag)
        print(f"OWNER_READY keys={keys}", flush=True)
        _wait_for(
            lambda: all(key not in list_samples() for key in keys),
            args.timeout,
            "client to clear the smoke sample keys",
        )
        print("OWNER_OBSERVED_SAMPLES_CLEARED", flush=True)
    finally:
        close_transfer_queue_owner()
        print("OWNER_CLOSED", flush=True)


def run_client(args) -> None:
    config = _config(args)
    store = TQFeatureStore.from_config(config)
    loader = TQFeatureDataLoader(store, batch_size=2, rank=0, world_size=1)
    try:
        iterator = iter(loader)
        batch = next(iterator)
        restored = batch.local_samples
        assert restored[0].hidden_states.tolist() == torch.arange(12).reshape(3, 4).tolist()
        assert restored[1].hidden_states.tolist() == (
            torch.arange(12).reshape(3, 4) + 1
        ).tolist()
        loader.clear_completed_batch(batch.global_keys)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("TQ Consumer did not stop after draining EOS")
        print(
            f"CLIENT_OK samples={len(restored)} shape={tuple(restored[0].hidden_states.shape)}",
            flush=True,
        )
    finally:
        store.close_local()
        print("CLIENT_CLOSED_LOCAL_ONLY", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("owner", "client"))
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--namespace", default="speco-drafter-smoke")
    parser.add_argument("--run-id", default="tq-smoke")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.role == "owner":
        run_owner(args)
    else:
        run_client(args)


if __name__ == "__main__":
    main()
