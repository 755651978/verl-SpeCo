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

# One-command standalone DSpark draft-model training. The launcher internally
# starts the hidden-state target vLLM and uses the
# Producer -> TransferQueue -> Consumer path.

project_name=verl_dspark_drafter
exp_name=qwen3_8b_dspark_separate_training

draft_train_gpus_per_node=8

MODEL_PATH=/path/to/Qwen3-8B
# Ordinary verl prompt Parquet is supported; target vLLM generates responses.
TRAIN_FILE=/path/to/train_file.parquet
DRAFTER_PATH=/path/to/vllm-compatible-dspark-drafter
DRAFT_CKPTS_DIR=/path/to/dspark_draft_checkpoints

PYTHON_BIN=${PYTHON_BIN:-python3}

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m verl_speco.standalone_tq_training_launcher \
    speco.draft_training.num_gpus_per_node=${draft_train_gpus_per_node} \
    speco.draft_training.nnodes=1 \
    speco.draft_training.standalone=True \
    data.train_files=${TRAIN_FILE} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.training.mode=offline \
    actor_rollout_ref.rollout.drafter.training.max_steps=10 \
    actor_rollout_ref.rollout.drafter.training.save_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.drafter.training.lr=1e-6 \
    actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.drafter.training.warmup_style=constant \
    actor_rollout_ref.rollout.drafter.training.use_logits=False \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    "$@"
