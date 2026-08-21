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

: "${RAY_ADDRESS:?Set RAY_ADDRESS to the running Ray head}"
: "${SPECO_TQ_RUN_ID:?Set SPECO_TQ_RUN_ID to the owner/consumer run id}"
: "${PRODUCER_INPUT_PATH:?Set PRODUCER_INPUT_PATH to prompt/response JSONL}"
: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to the target model id/path}"
: "${TARGET_MODEL_REVISION:?Set TARGET_MODEL_REVISION to a revision or checksum}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the tokenizer id/path}"
: "${TOKENIZER_FINGERPRINT:?Set TOKENIZER_FINGERPRINT to a verified fingerprint}"
: "${TARGET_LAYER_IDS:?Set TARGET_LAYER_IDS as a Hydra list, for example '[2,8,14,20,26]'}"
: "${VLLM_ENDPOINTS:?Set VLLM_ENDPOINTS as a Hydra list, for example '[http://node0:8000/v1]'}"
PYTHON_BIN=${PYTHON_BIN:-python3}
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter}
TQ_PARTITION_ID=${TQ_PARTITION_ID:-speco_drafter_features}
VLLM_MODEL=${VLLM_MODEL:-${TARGET_MODEL_PATH}}

exec "${PYTHON_BIN}" -m verl_speco.standalone_tq_producer \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=true \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.address="${RAY_ADDRESS}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.namespace="${TQ_NAMESPACE}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.partition_id="${TQ_PARTITION_ID}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id="${SPECO_TQ_RUN_ID}" \
  speco.standalone_tq_producer.input_path="${PRODUCER_INPUT_PATH}" \
  speco.standalone_tq_producer.target_model_id="${TARGET_MODEL_PATH}" \
  speco.standalone_tq_producer.target_model_revision="${TARGET_MODEL_REVISION}" \
  speco.standalone_tq_producer.tokenizer_path="${TOKENIZER_PATH}" \
  speco.standalone_tq_producer.tokenizer_fingerprint="${TOKENIZER_FINGERPRINT}" \
  speco.standalone_tq_producer.target_layer_ids="${TARGET_LAYER_IDS}" \
  speco.standalone_tq_producer.vllm_endpoints="${VLLM_ENDPOINTS}" \
  speco.standalone_tq_producer.vllm_model="${VLLM_MODEL}" \
  "$@"
