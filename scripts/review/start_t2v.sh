#!/usr/bin/env bash
# Serve evaluator debug bundles without modifying evaluator artifacts. The
# viewer maintains a hidden summary index at the REVIEW_DIR root.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${REVIEW_DIR:-}" ]]; then
  echo "start_t2v.sh: REVIEW_DIR is required; point it at a T2V debug directory" >&2
  echo "Example: REVIEW_DIR=./evaluation/model/run/t2v_debug ./scripts/review/start_t2v.sh" >&2
  exit 2
fi
if [[ "$REVIEW_DIR" != /* ]]; then
  REVIEW_DIR="$ROOT/${REVIEW_DIR#./}"
fi
if [[ ! -d "$REVIEW_DIR" ]]; then
  echo "start_t2v.sh: REVIEW_DIR does not exist or is not a directory: $REVIEW_DIR" >&2
  exit 2
fi
FRONTEND_DIR="${FRONTEND_DIR:-$ROOT/frontend/t2v_review}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8767}"
# VSCode/CloudML port forwarding usually needs HOST=0.0.0.0. Keep the
# localhost default for local-only use and print an actionable hint below.
# Video playback is opt-in. Review bundles remain useful when some sample IDs
# have no video, because the API lists cases from their detail manifests.
VIDEO_DIR="${VIDEO_DIR:-}"
video_args=()
if [[ -n "$VIDEO_DIR" ]]; then
  if [[ "$VIDEO_DIR" != /* ]]; then
    VIDEO_DIR="$ROOT/${VIDEO_DIR#./}"
  fi
  video_args=(--video-dir "$VIDEO_DIR")
fi
if [[ "$HOST" == "127.0.0.1" || "$HOST" == "localhost" ]]; then
  echo "start_t2v.sh: serving on $HOST:$PORT; for port forwarding, set HOST=0.0.0.0" >&2
fi
exec python3 -m textvbench.evaluators.t2v.review_api.server \
  --data "$REVIEW_DIR" --frontend "$FRONTEND_DIR" \
  --host "$HOST" --port "$PORT" "${video_args[@]}"
