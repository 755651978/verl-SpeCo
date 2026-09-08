#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
set -euo pipefail

# Edit these defaults, or override them before bash, for example:
# MODEL_PATH=/model/Qwen3-4B VLLM_DEVICES=0,1 TRAIN_DEVICES=2,3 bash "$0"
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-4B}
VLLM_DEVICES=${VLLM_DEVICES:-0,1}
TRAIN_DEVICES=${TRAIN_DEVICES:-2,3}
VLLM_TP=${VLLM_TP:-2}
TRAIN_NPROC=${TRAIN_NPROC:-2}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-weight-sync-smoke}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
BUCKET_SIZE_MB=${BUCKET_SIZE_MB:-256}
TRAIN_STEPS=${TRAIN_STEPS:-1}
LR=${LR:-0.01}
LOGPROB_ATOL=${LOGPROB_ATOL:-0.1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-600}
VLLM_READY_TIMEOUT_SECONDS=${VLLM_READY_TIMEOUT_SECONDS:-300}

if [[ "${MODEL_PATH}" == /path/to/* ]]; then
    echo "Set MODEL_PATH to a local dense Hugging Face model directory" >&2
    exit 2
fi
if ! [[ "${VLLM_TP}" =~ ^[1-9][0-9]*$ && "${TRAIN_NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "VLLM_TP and TRAIN_NPROC must be positive integers" >&2
    exit 2
fi
IFS=',' read -r -a VLLM_DEVICE_IDS <<< "${VLLM_DEVICES}"
IFS=',' read -r -a TRAIN_DEVICE_IDS <<< "${TRAIN_DEVICES}"
if (( ${#VLLM_DEVICE_IDS[@]} != VLLM_TP )); then
    echo "VLLM_DEVICES count must equal VLLM_TP" >&2
    exit 2
fi
if (( ${#TRAIN_DEVICE_IDS[@]} != TRAIN_NPROC )); then
    echo "TRAIN_DEVICES count must equal TRAIN_NPROC" >&2
    exit 2
fi

for device in "${VLLM_DEVICE_IDS[@]}"; do
    for train_device in "${TRAIN_DEVICE_IDS[@]}"; do
        if [[ "${device//[[:space:]]/}" == "${train_device//[[:space:]]/}" ]]; then
            echo "vLLM and FSDP2 must use different NPUs; duplicated device ${device}" >&2
            exit 2
        fi
    done
done

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting vLLM-Ascend: devices=${VLLM_DEVICES} tp=${VLLM_TP} port=${VLLM_PORT}"
# vLLM-Ascend 0.23 keeps upstream's accepted "nccl" backend name, then its
# platform patch resolves that registry entry to HCCLWeightTransferEngine.
VLLM_SERVER_DEV_MODE=1 \
VLLM_ASCEND_ENABLE_NZ=0 \
ASCEND_RT_VISIBLE_DEVICES="${VLLM_DEVICES}" \
vllm serve "${MODEL_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${VLLM_TP}" \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enforce-eager \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --load-format dummy \
    --weight-transfer-config '{"backend":"nccl"}' &
VLLM_PID=$!

python tools/wait_for_vllm_endpoints.py \
    --endpoints "[http://${VLLM_HOST}:${VLLM_PORT}/v1]" \
    --timeout-seconds "${VLLM_READY_TIMEOUT_SECONDS}"

echo "Starting FSDP2 sender: devices=${TRAIN_DEVICES} ranks=${TRAIN_NPROC}"
ASCEND_RT_VISIBLE_DEVICES="${TRAIN_DEVICES}" \
python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${TRAIN_NPROC}" \
    tools/fsdp2_vllm_weight_sync_smoke.py \
    --model "${MODEL_PATH}" \
    --endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --bucket-size-mb "${BUCKET_SIZE_MB}" \
    --train-steps "${TRAIN_STEPS}" \
    --lr "${LR}" \
    --logprob-atol "${LOGPROB_ATOL}" \
    --timeout "${TIMEOUT_SECONDS}"

echo "NPU_WEIGHT_SYNC_SMOKE_PASS"
