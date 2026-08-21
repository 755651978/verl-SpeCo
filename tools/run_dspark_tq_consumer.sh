#!/usr/bin/env bash
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
set -euo pipefail
set -x

# Standalone DSpark Consumer. Start Ray and verl_speco.tq_owner first, then run
# the Producer with the same Ray address, namespace, partition and run ID.
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
DRAFTER_PATH=${DRAFTER_PATH:-/path/to/dspark-drafter}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/dspark-tq-checkpoints}
TRAIN_DEVICES=${TRAIN_DEVICES:-0,1,2,3}
TRAIN_GPUS=${TRAIN_GPUS:-4}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-2}
MAX_STEPS=${MAX_STEPS:-1000}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter}
TQ_PARTITION_ID=${TQ_PARTITION_ID:-speco_drafter_features}
SPECO_TQ_RUN_ID=${SPECO_TQ_RUN_ID:-dspark-standalone-run}
PYTHON_BIN=${PYTHON_BIN:-python3}

CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONUNBUFFERED=1 \
exec "${PYTHON_BIN}" -m verl_speco.draft_train_launcher \
  speco.draft_training.nproc_per_node=${TRAIN_GPUS} \
  speco.draft_training.nnodes=1 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.rollout.drafter.enable=True \
  actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
  actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
  actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  actor_rollout_ref.rollout.drafter.training.mode=offline \
  actor_rollout_ref.rollout.drafter.training.feature_store.type=tq \
  actor_rollout_ref.rollout.drafter.training.feature_store.path=null \
  actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=False \
  actor_rollout_ref.rollout.drafter.training.feature_store.repeat=False \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=True \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.address=${RAY_ADDRESS} \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.namespace=${TQ_NAMESPACE} \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.partition_id=${TQ_PARTITION_ID} \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id=${SPECO_TQ_RUN_ID} \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.drop_last=True \
  actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=${BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.rollout.drafter.training.max_steps=${MAX_STEPS} \
  actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=${DSPARK_NUM_TARGET_LAYERS} \
  actor_rollout_ref.rollout.drafter.training.save_interval_steps=100 \
  actor_rollout_ref.rollout.drafter.training.lr=1e-5 \
  actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=50 \
  actor_rollout_ref.rollout.drafter.training.warmup_style=cosine \
  "$@"
