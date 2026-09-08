#!/usr/bin/env bash
# Start only the normally-loaded vLLM used by the weight-difference test.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-4B}
VLLM_DEVICES=${VLLM_DEVICES:-0,1}
VLLM_TP=${VLLM_TP:-2}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-weight-difference-test}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}

if [[ "${MODEL_PATH}" == /path/to/* ]]; then
    echo "Set MODEL_PATH to a local dense Hugging Face model directory" >&2
    exit 2
fi

export VLLM_SERVER_DEV_MODE=1
export VLLM_ASCEND_ENABLE_NZ=0
export ASCEND_RT_VISIBLE_DEVICES="${VLLM_DEVICES}"

exec vllm serve "${MODEL_PATH}" \
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
    --weight-transfer-config '{"backend":"nccl"}'
