# Ref2V Evaluator Review

This directory contains the bilingual, read-only audit interface for Ref2V
evaluator outputs. Its frontend and API are benchmark-specific and do not
mutate evaluation results.

## Start the service

Run the evaluator with `--debug-dir` before opening the Review UI. Then, from
the TextVBench repository root:

```bash
REVIEW_DIR=./evaluation/<model>/<run>/ref2v_debug \
RESULTS_DIR=./evaluation/<model>/<run>/ref2v \
VIDEO_DIR=./generated/<model>/<run>/videos \
./scripts/review/start_ref2v.sh
```

Open `http://127.0.0.1:8768`. Set `HOST=0.0.0.0` and forward `PORT` when the
process runs on a remote development machine.

`REVIEW_DIR` is required and may point either to the full debug root or to one
sample directory. `RESULTS_DIR` is optional but recommended because its public
metrics take precedence over the debug copy. Videos named `<sample_id>.mp4`
are discovered through `VIDEO_DIR`. For arbitrary names, set `VIDEO_MANIFEST`
to a JSONL file containing `sample_id` and `video_path` per row; relative video
paths are resolved from that manifest's directory. `PYTHON_BIN`, `HOST`,
`PORT`, and `FRONTEND_DIR` are also configurable.

The equivalent direct invocation is:

```bash
PYTHONPATH=src python3 -m textvbench.evaluators.ref2v.review_api.server \
  --data ./evaluation/<model>/<run>/ref2v_debug \
  --results-dir ./evaluation/<model>/<run>/ref2v \
  --video-dir ./generated/<model>/<run>/videos \
  --frontend frontend/ref2v_review \
  --port 8768
```

An installed package also provides the `textvbench-ref2v-review` command with
the same arguments.

For a non-canonical video filename, use a JSONL manifest and set
`VIDEO_MANIFEST` instead of relying on `<sample_id>.mp4` discovery:

```json
{"sample_id":"1190","video_path":"videos/custom-name.mp4"}
```

## Evidence shown

The case list supports search, status filters, metric sorting, and pagination.
A case exposes final metrics, reference images and OCR slots, sampled frames,
candidate filtering states, candidate reuse, Ref-ID routing, target matching
batches, VLM payloads/responses/retries, reading-order evidence, per-target
temporal evidence, and sanitized raw records. Single-reference cases explicitly
show that Ref-ID routing was skipped. The language switch supports Chinese and
English.

No-match frames are marked as excluded. Failed attempts before a successful
VLM retry remain visible without incorrectly marking the final case as failed.
The reading-order panel shows the reference sequence, matched predicted
sequence, missing-at-end effective sequence, score, and incorrect pairs.

Candidate tables distinguish active, confidence-filtered, duplicate-extra,
and capped-extra states. If one generated OCR box was selected for multiple
targets in a batch, every assigned slot ID is shown.

## Read-only API

The frontend uses these route families:

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Service health and indexed case count |
| `GET /api/cases` | Filtered, sorted, paginated case summaries |
| `GET /api/cases/<sample_id>` | Sanitized metrics and evidence index |
| `GET /api/cases/<sample_id>/frames/...` | Allowlisted frame JSON and VLM traces |
| `GET /api/cases/<sample_id>/references/...` | Allowlisted reference OCR JSON |
| `GET /media/cases/<sample_id>/...` | Reference/frame images and configured video |

The video endpoint supports HTTP Range requests. Unknown artifact names and
path traversal attempts are rejected.

## Security

The server only exposes allowlisted artifacts beneath the selected case. It
redacts credentials, endpoints, and machine-specific paths from JSON and uses
HTTP Range responses for video playback. The original debug files can still
contain private values. Do not publish the debug directory directly; use an
authenticated reverse proxy if this local review service is exposed to a
network.

Sanitization protects server configuration details; it does not remove text
or imagery that is part of the benchmark sample. Review the dataset content
before sharing a service publicly.
