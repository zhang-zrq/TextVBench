#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/checkpoints/Qwen3.8-27B}"
HF_BIN="${HF_BIN:-hf}"

if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  echo "Cannot find '${HF_BIN}'. Install it with: python -m pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$(dirname "${MODEL_DIR}")"
"${HF_BIN}" download "${MODEL_ID}" --local-dir "${MODEL_DIR}" "$@"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Download finished without config.json: ${MODEL_DIR}" >&2
  exit 1
fi
echo "VLM checkpoint ready: ${MODEL_DIR}"
