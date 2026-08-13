set -euo pipefail
set -x

# Run each stage in a separate shell: RUN_STAGE=master, vllm, then train.
# On Ascend replace CUDA_VISIBLE_DEVICES below with ASCEND_RT_VISIBLE_DEVICES.
RUN_STAGE=${RUN_STAGE:-train}
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen3-4B}
DATA_PATH=${DATA_PATH:-/path/to/token_replay.jsonl}
DRAFT_CKPTS_DIR=${DRAFT_CKPTS_DIR:-/path/to/draft_checkpoints}
VLLM_DEVICES=${VLLM_DEVICES:-0,1}
TRAIN_DEVICES=${TRAIN_DEVICES:-2,3,4,5}
VLLM_TP=${VLLM_TP:-2}
TRAIN_GPUS=${TRAIN_GPUS:-4}

export MOONCAKE_MASTER_SERVER=${MOONCAKE_MASTER_SERVER:-127.0.0.1:50051}
export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:8090/metadata}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-17179869184}
export MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE:-2147483648}

if [ "${RUN_STAGE}" = "master" ]; then
  exec mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=8090
fi

if [ "${RUN_STAGE}" = "vllm" ]; then
  CUDA_VISIBLE_DEVICES=${VLLM_DEVICES} exec vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "${VLLM_TP}" \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[1,9,17,25,33,36]}}}' \
    --kv-transfer-config '{"kv_connector":"SpeCoMooncakeHiddenStatesConnector","kv_connector_module_path":"verl_speco.integration.mooncake_hidden_states_connector","kv_role":"kv_producer"}' \
    --no-enable-chunked-prefill
fi

if [ "${RUN_STAGE}" != "train" ]; then
  echo "RUN_STAGE must be master, vllm, or train" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONUNBUFFERED=1 \
python3 -m verl_speco.draft_train_launcher \
  speco.draft_training.nproc_per_node=${TRAIN_GPUS} \
  speco.draft_training.nnodes=1 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.rollout.drafter.enable=True \
  actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
  actor_rollout_ref.rollout.drafter.checkpoint_path=${DRAFT_CKPTS_DIR} \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
  actor_rollout_ref.rollout.drafter.training.mode=offline \
  actor_rollout_ref.rollout.drafter.training.feature_store.type=jsonl_token_replay \
  actor_rollout_ref.rollout.drafter.training.feature_store.path=${DATA_PATH} \
  actor_rollout_ref.rollout.drafter.training.feature_store.shuffle=True \
  actor_rollout_ref.rollout.drafter.training.feature_store.repeat=True \
  actor_rollout_ref.rollout.drafter.training.target_feature_replay.backend=vllm_mooncake \
  actor_rollout_ref.rollout.drafter.training.target_feature_replay.vllm_endpoint=http://127.0.0.1:8000/v1 \
  actor_rollout_ref.rollout.drafter.training.target_feature_replay.on_generate=delete \
  actor_rollout_ref.rollout.drafter.training.target_feature_pipeline.enabled=True \
  actor_rollout_ref.rollout.drafter.training.target_feature_pipeline.concurrency=16 \
  actor_rollout_ref.rollout.drafter.training.target_feature_pipeline.transfer_concurrency=8 \
  actor_rollout_ref.rollout.drafter.training.target_feature_pipeline.producer_prefetch_depth=4 \
  actor_rollout_ref.rollout.drafter.training.target_feature_pipeline.prefetch_depth=2 \
  actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.drafter.training.max_steps=1000 \
  actor_rollout_ref.rollout.drafter.training.save_interval_steps=100 \
  actor_rollout_ref.rollout.drafter.training.lr=1e-5 \
  actor_rollout_ref.rollout.drafter.training.lr_warmup_steps=50 \
  actor_rollout_ref.rollout.drafter.training.warmup_style=cosine \
  "$@"
