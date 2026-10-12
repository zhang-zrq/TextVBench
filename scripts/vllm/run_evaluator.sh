#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
VLLM_LOG="${VLLM_LOG:-${PROJECT_ROOT}/output/vllm/server.log}"
export VLLM_HOST VLLM_PORT SERVED_MODEL_NAME

if (($# == 0)); then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  echo "example: $0 .venv/bin/textvbench evaluate t2v ..." >&2
  exit 2
fi
if [[ "${1}" == "--" ]]; then
  shift
fi
if (($# == 0)); then
  echo "missing evaluator command" >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to manage the local vLLM process." >&2
  exit 1
fi

mkdir -p "$(dirname "${VLLM_LOG}")"
vllm_pid=""
stop_vllm() {
  if [[ -n "${vllm_pid}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
    kill -TERM -- "-${vllm_pid}" 2>/dev/null || kill -TERM "${vllm_pid}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${vllm_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${vllm_pid}" 2>/dev/null; then
      kill -KILL -- "-${vllm_pid}" 2>/dev/null || kill -KILL "${vllm_pid}" 2>/dev/null || true
    fi
  fi
  [[ -z "${vllm_pid}" ]] || wait "${vllm_pid}" 2>/dev/null || true
}
trap stop_vllm EXIT INT TERM

echo "Starting local vLLM; log=${VLLM_LOG}"
setsid "${SCRIPT_DIR}/start_server.sh" >"${VLLM_LOG}" 2>&1 &
vllm_pid="$!"
VLLM_PID="${vllm_pid}" "${SCRIPT_DIR}/wait_ready.sh" || {
  tail -n 200 "${VLLM_LOG}" >&2 || true
  exit 1
}

export TEXTVBENCH_VLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export TEXTVBENCH_VLM_MODEL="${SERVED_MODEL_NAME}"
export TEXTVBENCH_VLM_API_KEY="${TEXTVBENCH_VLM_API_KEY:-EMPTY}"
"$@"
