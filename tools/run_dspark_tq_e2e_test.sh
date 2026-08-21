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
# Run the real standalone TQ Producer and Consumer in one end-to-end test.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

: "${RAY_ADDRESS:?Set RAY_ADDRESS to the running Ray head}"
: "${MODEL_PATH:?Set MODEL_PATH to the target model directory}"
: "${DRAFTER_PATH:?Set DRAFTER_PATH to the DSpark drafter directory}"
: "${PRODUCER_INPUT_PATH:?Set PRODUCER_INPUT_PATH to prompt/response JSONL}"
: "${TARGET_MODEL_REVISION:?Set TARGET_MODEL_REVISION to a revision or checksum}"
: "${TOKENIZER_FINGERPRINT:?Set TOKENIZER_FINGERPRINT to a verified fingerprint}"
: "${TARGET_LAYER_IDS:?Set TARGET_LAYER_IDS as a Hydra list, for example '[2,8,14,20,26]'}"
: "${VLLM_ENDPOINTS:?Set VLLM_ENDPOINTS as a Hydra list, for example '[http://node0:8000/v1]'}"

PYTHON_BIN=${PYTHON_BIN:-python3}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MODEL_PATH}}
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter}
TQ_PARTITION_ID=${TQ_PARTITION_ID:-speco_drafter_features}
SPECO_TQ_RUN_ID=${SPECO_TQ_RUN_ID:-dspark-e2e-$(date +%Y%m%d-%H%M%S)-$$}
TRAIN_DEVICES=${TRAIN_DEVICES:-0}
TRAIN_GPUS=${TRAIN_GPUS:-1}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/tmp/speco-dspark-tq-e2e-${SPECO_TQ_RUN_ID}}
E2E_TIMEOUT_SECONDS=${E2E_TIMEOUT_SECONDS:-1800}

if [[ "${TQ_PARTITION_ID}" != "speco_drafter_features" ]]; then
  echo "TQ_PARTITION_ID must be speco_drafter_features for protocol v1" >&2
  exit 2
fi
if [[ ! -f "${PRODUCER_INPUT_PATH}" ]]; then
  echo "Producer input does not exist: ${PRODUCER_INPUT_PATH}" >&2
  exit 2
fi
input_samples=$(awk 'NF { count++ } END { print count + 0 }' "${PRODUCER_INPUT_PATH}")
global_batch_size=$((TRAIN_GPUS * BATCH_SIZE_PER_GPU))
if (( input_samples < global_batch_size )); then
  echo "Producer input has ${input_samples} non-empty records, but one Consumer global batch needs ${global_batch_size}" >&2
  exit 2
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/speco-tq-e2e.XXXXXX")
owner_pid=""
consumer_pid=""
producer_pid=""

cleanup() {
  local pid
  for pid in "${producer_pid}" "${consumer_pid}" "${owner_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT INT TERM

echo "[1/4] Starting TQ owner run_id=${SPECO_TQ_RUN_ID}"
RAY_ADDRESS="${RAY_ADDRESS}" \
TQ_NAMESPACE="${TQ_NAMESPACE}" \
TQ_PARTITION_ID="${TQ_PARTITION_ID}" \
SPECO_TQ_RUN_ID="${SPECO_TQ_RUN_ID}" \
  bash tools/run_dspark_tq_owner.sh &
owner_pid=$!

echo "[2/4] Starting real DSpark Consumer"
(
  set +e
  MODEL_PATH="${MODEL_PATH}" \
  DRAFTER_PATH="${DRAFTER_PATH}" \
  DRAFT_CKPTS_DIR="${DRAFT_CKPTS_DIR}" \
  TRAIN_DEVICES="${TRAIN_DEVICES}" \
  TRAIN_GPUS="${TRAIN_GPUS}" \
  BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU}" \
  MAX_STEPS=0 \
  DSPARK_NUM_TARGET_LAYERS="${DSPARK_NUM_TARGET_LAYERS}" \
  RAY_ADDRESS="${RAY_ADDRESS}" \
  TQ_NAMESPACE="${TQ_NAMESPACE}" \
  TQ_PARTITION_ID="${TQ_PARTITION_ID}" \
  SPECO_TQ_RUN_ID="${SPECO_TQ_RUN_ID}" \
    bash tools/run_dspark_tq_consumer.sh "$@"
  echo "$?" > "${work_dir}/consumer.status"
) &
consumer_pid=$!

echo "[3/4] Starting real vLLM-backed Producer"
(
  set +e
  PYTHON_BIN="${PYTHON_BIN}" \
  RAY_ADDRESS="${RAY_ADDRESS}" \
  TQ_NAMESPACE="${TQ_NAMESPACE}" \
  TQ_PARTITION_ID="${TQ_PARTITION_ID}" \
  SPECO_TQ_RUN_ID="${SPECO_TQ_RUN_ID}" \
  PRODUCER_INPUT_PATH="${PRODUCER_INPUT_PATH}" \
  TARGET_MODEL_PATH="${TARGET_MODEL_PATH}" \
  TARGET_MODEL_REVISION="${TARGET_MODEL_REVISION}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  TOKENIZER_FINGERPRINT="${TOKENIZER_FINGERPRINT}" \
  TARGET_LAYER_IDS="${TARGET_LAYER_IDS}" \
  VLLM_ENDPOINTS="${VLLM_ENDPOINTS}" \
    bash examples/run_dspark_tq_producer.sh
  echo "$?" > "${work_dir}/producer.status"
) &
producer_pid=$!

started_at=${SECONDS}
while [[ ! -f "${work_dir}/producer.status" || ! -f "${work_dir}/consumer.status" ]]; do
  if ! kill -0 "${owner_pid}" 2>/dev/null; then
    echo "TQ owner exited before the end-to-end test completed" >&2
    exit 1
  fi
  if (( SECONDS - started_at >= E2E_TIMEOUT_SECONDS )); then
    echo "E2E timed out after ${E2E_TIMEOUT_SECONDS} seconds" >&2
    exit 124
  fi
  if [[ -f "${work_dir}/producer.status" ]]; then
    producer_status=$(<"${work_dir}/producer.status")
    if [[ "${producer_status}" -ne 0 ]]; then
      echo "Producer failed with exit code ${producer_status}" >&2
      exit "${producer_status}"
    fi
  fi
  if [[ -f "${work_dir}/consumer.status" ]]; then
    consumer_status=$(<"${work_dir}/consumer.status")
    if [[ "${consumer_status}" -ne 0 ]]; then
      echo "Consumer failed with exit code ${consumer_status}" >&2
      exit "${consumer_status}"
    fi
  fi
  sleep 1
done

producer_status=$(<"${work_dir}/producer.status")
consumer_status=$(<"${work_dir}/consumer.status")
if [[ "${producer_status}" -ne 0 || "${consumer_status}" -ne 0 ]]; then
  echo "E2E failed: producer=${producer_status} consumer=${consumer_status}" >&2
  exit 1
fi

wait "${producer_pid}"
producer_pid=""
wait "${consumer_pid}"
consumer_pid=""

echo "[4/4] Producer published EOS and Consumer drained the run; stopping owner"
kill "${owner_pid}" 2>/dev/null || true
wait "${owner_pid}" 2>/dev/null || true
owner_pid=""
trap - EXIT INT TERM
rm -rf -- "${work_dir}"

echo "DSPARK_TQ_E2E_TEST_OK run_id=${SPECO_TQ_RUN_ID} checkpoints=${DRAFT_CKPTS_DIR}"
