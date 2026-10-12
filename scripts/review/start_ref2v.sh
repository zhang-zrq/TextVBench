#!/usr/bin/env bash
# Serve Ref2V evaluator evidence without modifying evaluator artifacts. The
# viewer maintains a hidden summary index at the REVIEW_DIR root.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${REVIEW_DIR:-}" ]]; then
  echo "start_ref2v.sh: REVIEW_DIR is required" >&2
  echo "Example: REVIEW_DIR=./evaluation/model/run/ref2v_debug RESULTS_DIR=./evaluation/model/run/ref2v ./scripts/review/start_ref2v.sh" >&2
  exit 2
fi

resolve_from_root() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s' "$value"
  else
    printf '%s/%s' "$ROOT" "${value#./}"
  fi
}

REVIEW_DIR="$(resolve_from_root "$REVIEW_DIR")"
FRONTEND_DIR="$(resolve_from_root "${FRONTEND_DIR:-frontend/ref2v_review}")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8768}"

if [[ ! -d "$REVIEW_DIR" ]]; then
  echo "start_ref2v.sh: REVIEW_DIR is not a directory: $REVIEW_DIR" >&2
  exit 2
fi
if [[ ! -f "$FRONTEND_DIR/index.html" ]]; then
  echo "start_ref2v.sh: frontend not found: $FRONTEND_DIR" >&2
  exit 2
fi

args=(--data "$REVIEW_DIR" --frontend "$FRONTEND_DIR" --host "$HOST" --port "$PORT")

if [[ -n "${RESULTS_DIR:-}" ]]; then
  RESULTS_DIR="$(resolve_from_root "$RESULTS_DIR")"
  [[ -d "$RESULTS_DIR" ]] || { echo "start_ref2v.sh: RESULTS_DIR is not a directory: $RESULTS_DIR" >&2; exit 2; }
  args+=(--results-dir "$RESULTS_DIR")
fi
if [[ -n "${VIDEO_DIR:-}" ]]; then
  VIDEO_DIR="$(resolve_from_root "$VIDEO_DIR")"
  [[ -d "$VIDEO_DIR" ]] || { echo "start_ref2v.sh: VIDEO_DIR is not a directory: $VIDEO_DIR" >&2; exit 2; }
  args+=(--video-dir "$VIDEO_DIR")
fi
if [[ -n "${VIDEO_MANIFEST:-}" ]]; then
  VIDEO_MANIFEST="$(resolve_from_root "$VIDEO_MANIFEST")"
  [[ -f "$VIDEO_MANIFEST" ]] || { echo "start_ref2v.sh: VIDEO_MANIFEST is not a file: $VIDEO_MANIFEST" >&2; exit 2; }
  args+=(--video-manifest "$VIDEO_MANIFEST")
fi

if [[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]]; then
  echo "Ref2V review is local-only; set HOST=0.0.0.0 when port forwarding is required." >&2
fi

exec "$PYTHON_BIN" -m textvbench.evaluators.ref2v.review_api.server "${args[@]}"
