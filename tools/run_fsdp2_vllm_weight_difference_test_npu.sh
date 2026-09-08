#!/usr/bin/env bash
# Run only the FSDP2 side of the weight-difference test.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/nas/disk1/Qwen3-4B}
TRAIN_DEVICES=${TRAIN_DEVICES:-1,2}
TRAIN_NPROC=${TRAIN_NPROC:-2}
VLLM_ENDPOINT=${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-weight-difference-test}
POD_IP=${POD_IP:-$(hostname -I | awk '{print $1}')}
MASTER_ADDRESS=${MASTER_ADDRESS:-${POD_IP}}
BUCKET_SIZE_MB=${BUCKET_SIZE_MB:-256}
LOGPROB_ATOL=${LOGPROB_ATOL:-0.4}
RANDOM_WEIGHT_SCALE=${RANDOM_WEIGHT_SCALE:-0.01}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600}
USE_RAY=${USE_RAY:-1}
CONNECT_ONLY=${CONNECT_ONLY:-1}
PROCESS_GROUP_LAYOUT=${PROCESS_GROUP_LAYOUT:-verl-composite}
PROCESS_GROUP_INIT=${PROCESS_GROUP_INIT:-verl-ray}
INSTALL_VERL_NPU_VLLM_COMPAT=${INSTALL_VERL_NPU_VLLM_COMPAT:-1}
INITIALIZE_VERL_WORKER_BASE=${INITIALIZE_VERL_WORKER_BASE:-1}
LOG_FILE=${LOG_FILE:-fast_test.log}

export ASCEND_RT_VISIBLE_DEVICES="${TRAIN_DEVICES}"

common_args=(
    --model "${MODEL_PATH}"
    --endpoint "${VLLM_ENDPOINT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --master-address "${MASTER_ADDRESS}"
    --bucket-size-mb "${BUCKET_SIZE_MB}"
    --logprob-atol "${LOGPROB_ATOL}"
    --random-weight-test
    --random-weight-scale "${RANDOM_WEIGHT_SCALE}"
    --timeout "${TIMEOUT_SECONDS}"
    --process-group-layout "${PROCESS_GROUP_LAYOUT}"
    --process-group-init "${PROCESS_GROUP_INIT}"
)

if [[ "${CONNECT_ONLY}" == "1" ]]; then
    common_args+=(--connect-only)
fi
if [[ "${INSTALL_VERL_NPU_VLLM_COMPAT}" == "1" ]]; then
    common_args+=(--install-verl-npu-vllm-compat)
fi
if [[ "${INITIALIZE_VERL_WORKER_BASE}" == "1" ]]; then
    common_args+=(--initialize-verl-worker-base)
fi

if [[ "${USE_RAY}" == "1" ]]; then
    echo "Running external-vLLM connect smoke; log: ${LOG_FILE}"
    if python tools/fsdp2_vllm_weight_sync_smoke.py \
        --ray \
        --ray-num-workers "${TRAIN_NPROC}" \
        "${common_args[@]}" >"${LOG_FILE}" 2>&1; then
        echo "Smoke test passed; log: ${LOG_FILE}"
    else
        status=$?
        echo "Smoke test failed with status ${status}; log: ${LOG_FILE}" >&2
        exit "${status}"
    fi
else
    echo "Running torchrun external-vLLM connect smoke; log: ${LOG_FILE}"
    if python -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${TRAIN_NPROC}" \
        tools/fsdp2_vllm_weight_sync_smoke.py \
        "${common_args[@]}" >"${LOG_FILE}" 2>&1; then
        echo "Smoke test passed; log: ${LOG_FILE}"
    else
        status=$?
        echo "Smoke test failed with status ${status}; log: ${LOG_FILE}" >&2
        exit "${status}"
    fi
fi
