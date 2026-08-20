#!/usr/bin/env bash
# Exercise the real standalone DSpark Consumer with delayed synthetic TQ data.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the target model directory}"
: "${DRAFTER_PATH:?Set DRAFTER_PATH to the DSpark drafter directory}"

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter}
TQ_PARTITION_ID=${TQ_PARTITION_ID:-speco_drafter_features}
SPECO_TQ_RUN_ID=${SPECO_TQ_RUN_ID:-dspark-consumer-test-$$}
TRAIN_DEVICES=${TRAIN_DEVICES:-0}
TRAIN_GPUS=${TRAIN_GPUS:-1}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}
NUM_BATCHES=${NUM_BATCHES:-3}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-64}
DSPARK_NUM_TARGET_LAYERS=${DSPARK_NUM_TARGET_LAYERS:-5}
INITIAL_DELAY_SECONDS=${INITIAL_DELAY_SECONDS:-5}
BATCH_INTERVAL_SECONDS=${BATCH_INTERVAL_SECONDS:-5}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/tmp/speco-dspark-tq-consumer-test-${SPECO_TQ_RUN_ID}}

owner_pid=""
producer_pid=""

cleanup() {
  if [[ -n "${producer_pid}" ]] && kill -0 "${producer_pid}" 2>/dev/null; then
    kill "${producer_pid}" 2>/dev/null || true
    wait "${producer_pid}" 2>/dev/null || true
  fi
  if [[ -n "${owner_pid}" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    kill "${owner_pid}" 2>/dev/null || true
    wait "${owner_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[1/4] Starting TQ owner run_id=${SPECO_TQ_RUN_ID}"
RAY_ADDRESS="${RAY_ADDRESS}" TQ_NAMESPACE="${TQ_NAMESPACE}" \
TQ_PARTITION_ID="${TQ_PARTITION_ID}" SPECO_TQ_RUN_ID="${SPECO_TQ_RUN_ID}" \
  bash tools/run_dspark_tq_owner.sh &
owner_pid=$!

echo "[2/4] Starting delayed synthetic Producer"
"${PYTHON_BIN}" tools/tq_delayed_test_producer.py \
  --model-path "${MODEL_PATH}" \
  --ray-address "${RAY_ADDRESS}" \
  --namespace "${TQ_NAMESPACE}" \
  --partition-id "${TQ_PARTITION_ID}" \
  --run-id "${SPECO_TQ_RUN_ID}" \
  --world-size "${TRAIN_GPUS}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --num-batches "${NUM_BATCHES}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --num-target-layers "${DSPARK_NUM_TARGET_LAYERS}" \
  --initial-delay "${INITIAL_DELAY_SECONDS}" \
  --batch-interval "${BATCH_INTERVAL_SECONDS}" &
producer_pid=$!

echo "[3/4] Running the real standalone Consumer; it should wait between batches"
MODEL_PATH="${MODEL_PATH}" \
DRAFTER_PATH="${DRAFTER_PATH}" \
DRAFT_CKPTS_DIR="${DRAFT_CKPTS_DIR}" \
TRAIN_DEVICES="${TRAIN_DEVICES}" \
TRAIN_GPUS="${TRAIN_GPUS}" \
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU}" \
MAX_STEPS="$((NUM_BATCHES + 1))" \
DSPARK_NUM_TARGET_LAYERS="${DSPARK_NUM_TARGET_LAYERS}" \
RAY_ADDRESS="${RAY_ADDRESS}" \
TQ_NAMESPACE="${TQ_NAMESPACE}" \
TQ_PARTITION_ID="${TQ_PARTITION_ID}" \
SPECO_TQ_RUN_ID="${SPECO_TQ_RUN_ID}" \
  bash tools/run_dspark_tq_consumer.sh "$@"

wait "${producer_pid}"
producer_pid=""

echo "[4/4] Consumer observed EOS and exited; stopping this test's TQ owner"
kill "${owner_pid}" 2>/dev/null || true
wait "${owner_pid}" 2>/dev/null || true
owner_pid=""
trap - EXIT INT TERM

echo "DSPARK_TQ_CONSUMER_TEST_OK run_id=${SPECO_TQ_RUN_ID} batches=${NUM_BATCHES}"
