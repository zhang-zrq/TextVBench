#!/usr/bin/env bash
set -Eeuo pipefail

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-5}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://${VLLM_HOST}:${VLLM_PORT}/health}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to check vLLM health." >&2
  exit 1
fi
if ! [[ "${STARTUP_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STARTUP_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
until curl --fail --silent --show-error "${VLLM_HEALTH_URL}" >/dev/null 2>&1; do
  if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming healthy." >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "vLLM was not healthy within ${STARTUP_TIMEOUT_SECONDS}s: ${VLLM_HEALTH_URL}" >&2
    exit 1
  fi
  sleep "${POLL_INTERVAL_SECONDS}"
done
echo "vLLM is ready: ${VLLM_HEALTH_URL}"
