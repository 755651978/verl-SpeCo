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
INITIALIZE_ACTOR_ROLLOUT_REF_BASE=${INITIALIZE_ACTOR_ROLLOUT_REF_BASE:-1}
USE_WORKER_DICT=${USE_WORKER_DICT:-1}
USE_RAY_WORKER_GROUP=${USE_RAY_WORKER_GROUP:-1}
USE_VERL_NPU_VLLM_COMPAT_MIXIN=${USE_VERL_NPU_VLLM_COMPAT_MIXIN:-1}
USE_DRAFT_WEIGHT_PUBLISH_MIXIN=${USE_DRAFT_WEIGHT_PUBLISH_MIXIN:-1}
USE_FULL_HYDRA_WORKER_CONFIG=${USE_FULL_HYDRA_WORKER_CONFIG:-1}
USE_RESOURCE_POOL_MANAGER=${USE_RESOURCE_POOL_MANAGER:-1}
USE_TASK_RUNNER_OUTER_ACTOR=${USE_TASK_RUNNER_OUTER_ACTOR:-1}
HIDE_DRAFTER_FROM_WORKER_CONFIG=${HIDE_DRAFTER_FROM_WORKER_CONFIG:-1}
# Keep this permanently empty for the controlled RPC comparison. The preceding
# run() path passed with this setting; only the RPC entry changes in this round.
FULL_HYDRA_EMPTY_ACTOR_PROFILER=${FULL_HYDRA_EMPTY_ACTOR_PROFILER:-0}
CONNECT_BEFORE_MODEL=${CONNECT_BEFORE_MODEL:-1}
INSTALL_ROLLOUT_RUNTIME=${INSTALL_ROLLOUT_RUNTIME:-1}
INSTALL_OLDLOGPROB_RUNTIME=${INSTALL_OLDLOGPROB_RUNTIME:-1}
WORKING_HCCL_INIT=${WORKING_HCCL_INIT:-true}
HCCL_PEER_WAIT_MODE=${HCCL_PEER_WAIT_MODE:-marker}
HCCL_INIT_IMPLEMENTATION=${HCCL_INIT_IMPLEMENTATION:-production}
LOG_FILE=${LOG_FILE:-fast_test.log}

export ASCEND_RT_VISIBLE_DEVICES="${TRAIN_DEVICES}"
export SPECO_CONNECT_SMOKE_INSTALL_ROLLOUT_RUNTIME="${INSTALL_ROLLOUT_RUNTIME}"
export SPECO_CONNECT_SMOKE_INSTALL_OLDLOGPROB_RUNTIME="${INSTALL_OLDLOGPROB_RUNTIME}"
export SPECO_CONNECT_SMOKE_PEER_WAIT_MODE="${HCCL_PEER_WAIT_MODE}"
if [[ "${WORKING_HCCL_INIT}" == "true" ]]; then
    export SPECO_CONNECT_SMOKE_SET_DEVICE_BEFORE_HCCL=1
else
    export SPECO_CONNECT_SMOKE_SET_DEVICE_BEFORE_HCCL=0
fi

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
    --working-hccl-init "${WORKING_HCCL_INIT}"
    --hccl-init-implementation "${HCCL_INIT_IMPLEMENTATION}"
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
if [[ "${INITIALIZE_ACTOR_ROLLOUT_REF_BASE}" == "1" ]]; then
    common_args+=(--initialize-actor-rollout-ref-base)
fi
if [[ "${USE_WORKER_DICT}" == "1" ]]; then
    common_args+=(--use-worker-dict)
fi
if [[ "${USE_RAY_WORKER_GROUP}" == "1" ]]; then
    common_args+=(--use-ray-worker-group)
fi
if [[ "${USE_VERL_NPU_VLLM_COMPAT_MIXIN}" == "1" ]]; then
    common_args+=(--use-verl-npu-vllm-compat-mixin)
fi
if [[ "${USE_DRAFT_WEIGHT_PUBLISH_MIXIN}" == "1" ]]; then
    common_args+=(--use-draft-weight-publish-mixin)
fi
if [[ "${USE_FULL_HYDRA_WORKER_CONFIG}" == "1" ]]; then
    common_args+=(--use-full-hydra-worker-config)
fi
if [[ "${USE_RESOURCE_POOL_MANAGER}" == "1" ]]; then
    common_args+=(--use-resource-pool-manager)
fi
if [[ "${USE_TASK_RUNNER_OUTER_ACTOR}" == "1" ]]; then
    common_args+=(--use-task-runner-outer-actor)
fi
if [[ "${HIDE_DRAFTER_FROM_WORKER_CONFIG}" == "1" ]]; then
    common_args+=(--hide-drafter-from-worker-config)
fi
if [[ "${FULL_HYDRA_EMPTY_ACTOR_PROFILER}" == "1" ]]; then
    common_args+=(--full-hydra-empty-actor-profiler)
fi
if [[ "${CONNECT_BEFORE_MODEL}" == "1" ]]; then
    common_args+=(--connect-before-model)
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
