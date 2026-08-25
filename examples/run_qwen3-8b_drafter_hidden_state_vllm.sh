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

# Start only the target-model vLLM used by the standalone TQ Producer.
# Run this script in its own terminal before starting the training script.
#
# Ascend example:
#   DEVICE_ENV=ASCEND_RT_VISIBLE_DEVICES VLLM_DEVICES_0=0,1 VLLM_DEVICES_1=2,3 VLLM_TP=2 \
#     bash examples/run_qwen3-8b_drafter_hidden_state_vllm.sh
# CUDA example:
#   DEVICE_ENV=CUDA_VISIBLE_DEVICES VLLM_DEVICES_0=0,1 VLLM_DEVICES_1=2,3 VLLM_TP=2 \
#     bash examples/run_qwen3-8b_drafter_hidden_state_vllm.sh

MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-8B}
DEVICE_ENV=${DEVICE_ENV:-ASCEND_RT_VISIBLE_DEVICES}
VLLM_DEVICES_0=${VLLM_DEVICES_0:-0}
VLLM_DEVICES_1=${VLLM_DEVICES_1:-1}
VLLM_TP=${VLLM_TP:-1}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT_0=${VLLM_PORT_0:-8000}
VLLM_PORT_1=${VLLM_PORT_1:-8001}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
# Auxiliary training layers followed by the target model's final hidden-state
# layer. Keep the auxiliary prefix aligned with DSPARK_TARGET_LAYER_IDS in the
# standalone training script. DSpark L1 loss consumes the final entry.
VLLM_HIDDEN_STATE_LAYER_IDS=${VLLM_HIDDEN_STATE_LAYER_IDS:-'[1,9,17,25,33,36]'}
HIDDEN_STATES_DIR=${HIDDEN_STATES_DIR:-/tmp/speco-vllm-hidden-states}

mkdir -p "${HIDDEN_STATES_DIR}/service-0" "${HIDDEN_STATES_DIR}/service-1"

SPECULATIVE_CONFIG=$(printf '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":%s}}}' "${VLLM_HIDDEN_STATE_LAYER_IDS}")

start_vllm() {
    local devices=$1
    local port=$2
    local hidden_states_dir=$3
    shift 3
    local kv_transfer_config
    kv_transfer_config=$(printf '{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"%s","use_synchronization_lock":true}}' "${hidden_states_dir}")
    env "${DEVICE_ENV}=${devices}" vllm serve "${MODEL_PATH}" \
        --host "${VLLM_HOST}" \
        --port "${port}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --speculative-config "${SPECULATIVE_CONFIG}" \
        --kv-transfer-config "${kv_transfer_config}" \
        --no-enable-chunked-prefill \
        "$@" &
    STARTED_PID=$!
}

PID_0=""
PID_1=""
cleanup() {
    [[ -n "${PID_0}" ]] && kill "${PID_0}" 2>/dev/null || true
    [[ -n "${PID_1}" ]] && kill "${PID_1}" 2>/dev/null || true
    [[ -n "${PID_0}" ]] && wait "${PID_0}" 2>/dev/null || true
    [[ -n "${PID_1}" ]] && wait "${PID_1}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_vllm "${VLLM_DEVICES_0}" "${VLLM_PORT_0}" "${HIDDEN_STATES_DIR}/service-0" "$@"
PID_0=${STARTED_PID}
start_vllm "${VLLM_DEVICES_1}" "${VLLM_PORT_1}" "${HIDDEN_STATES_DIR}/service-1" "$@"
PID_1=${STARTED_PID}

echo "VLLM_SERVICES_STARTED pid_0=${PID_0} endpoint_0=http://${VLLM_HOST}:${VLLM_PORT_0}/v1 pid_1=${PID_1} endpoint_1=http://${VLLM_HOST}:${VLLM_PORT_1}/v1"
set +e
wait -n "${PID_0}" "${PID_1}"
status=$?
set -e
exit "${status}"
