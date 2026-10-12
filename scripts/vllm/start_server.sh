#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/Qwen3.8-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Invalid VLM checkpoint: ${MODEL_PATH}" >&2
  echo "Run scripts/vllm/download_model.sh or set MODEL_PATH." >&2
  exit 1
fi
if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "Cannot find '${VLLM_BIN}'. Install vLLM or set VLLM_BIN." >&2
  exit 1
fi
if ! [[ "${VLLM_PORT}" =~ ^[0-9]+$ ]] || ((VLLM_PORT < 1 || VLLM_PORT > 65535)); then
  echo "VLLM_PORT must be between 1 and 65535: ${VLLM_PORT}" >&2
  exit 2
fi
if ! [[ "${VLLM_TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLLM_TP_SIZE must be a positive integer: ${VLLM_TP_SIZE}" >&2
  exit 2
fi

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  --gdn-prefill-backend triton \
  --enable-prefix-caching \
  "$@"
