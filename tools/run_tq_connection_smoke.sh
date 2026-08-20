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

# A Ray head must already be running. This script starts only the two smoke
# roles: an owner that also publishes two synthetic samples, and a client that
# reads, validates, clears, and observes EOS.
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379}
TQ_NAMESPACE=${TQ_NAMESPACE:-speco-drafter-smoke}
SPECO_TQ_RUN_ID=${SPECO_TQ_RUN_ID:-tq-smoke-$$}
SMOKE_TIMEOUT_SECONDS=${SMOKE_TIMEOUT_SECONDS:-60}

owner_pid=""

cleanup() {
  if [[ -n "${owner_pid}" ]] && kill -0 "${owner_pid}" 2>/dev/null; then
    kill "${owner_pid}" 2>/dev/null || true
    wait "${owner_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" tools/tq_connection_smoke.py owner \
  --ray-address "${RAY_ADDRESS}" \
  --namespace "${TQ_NAMESPACE}" \
  --run-id "${SPECO_TQ_RUN_ID}" \
  --timeout "${SMOKE_TIMEOUT_SECONDS}" &
owner_pid=$!

"${PYTHON_BIN}" tools/tq_connection_smoke.py client \
  --ray-address "${RAY_ADDRESS}" \
  --namespace "${TQ_NAMESPACE}" \
  --run-id "${SPECO_TQ_RUN_ID}" \
  --timeout "${SMOKE_TIMEOUT_SECONDS}"

wait "${owner_pid}"
owner_pid=""
trap - EXIT INT TERM

echo "TQ_CONNECTION_SMOKE_OK run_id=${SPECO_TQ_RUN_ID}"
