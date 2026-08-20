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

: "${RAY_ADDRESS:?Set RAY_ADDRESS to the running Ray head, for example 10.0.0.1:6379}"
: "${SPECO_TQ_RUN_ID:?Set SPECO_TQ_RUN_ID to a unique pipeline run id}"
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter}
TQ_PARTITION_ID=${TQ_PARTITION_ID:-speco_drafter_features}

python -m verl_speco.tq_owner \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.address="${RAY_ADDRESS}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.namespace="${TQ_NAMESPACE}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.partition_id="${TQ_PARTITION_ID}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id="${SPECO_TQ_RUN_ID}" \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.backend.storage_backend=SimpleStorage
