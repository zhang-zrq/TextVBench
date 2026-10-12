# TextVBench-Ref2V I/O Specification

This document defines the stable contract between a video-generation job and
the Ref2V evaluator. The release benchmark is the only ground-truth
input visible to benchmark users. Intermediate source manifests are not
evaluator inputs.

## 1. Ground-Truth Dataset

The evaluator accepts either the benchmark directory or its JSONL file:

```text
data/Ref2V/
├── video_prompt_dataset.jsonl
└── imgs/
    ├── 2024_4_10_0_4_95a248d7c9c347b7ab954e5b8d93004f-8.png
    └── ...
```

Each JSONL row has this shape:

```json
{
  "sample_id": "0001",
  "prompt": "...",
  "reference_images": [
    {
      "ref_id": "R01",
      "filename": "2024_4_10_0_4_95a248d7c9c347b7ab954e5b8d93004f-8.png"
    }
  ]
}
```

Rules:

- `sample_id` is a four-digit sequential string (`0001`, `0002`, ...), unique
  in the formal benchmark JSONL. It is assigned by the sampling step and is not
  changed by the evaluator.
- `reference_images[].filename` is resolved as
  `data/Ref2V/imgs/<filename>`. It is not an absolute path and does not
  use `image_root`.
- `ref_id` is unique within one sample and uses `R01`, `R02`, ... .
- No source, category, difficulty, provenance-path, or internal merged-ID fields
  are required or published in per-sample records.
- All visible OCR text in the reference image, including status-bar text, is
  discovered at evaluator runtime and becomes a slot.
- Before a reference image is sent to either VLM stage, the evaluator adds a
  white upper-left `Rxx` label band and resizes the resulting image
  proportionally so its longest edge is at most 256 pixels. OCR itself still
  runs on the original, unmodified reference image.

## 2. Generated-Video Directory

Keep generated videos outside the immutable release dataset. A recommended
generation-run layout is:

```text
generated/
└── <model_name>/
    └── <generation_run_id>/
        ├── videos/
        │   ├── 0001.mp4
        │   ├── 0002.mp4
        │   └── ...
        └── generation_manifest.json        # optional provenance
```

The canonical batch filename is exactly the four-digit `sample_id` plus the
`.mp4` suffix:

```text
sample_id:  0001
video:      videos/0001.mp4

sample_id:  1190
video:      videos/1190.mp4
```

Do not use the reference image filename, append an image suffix, put the model
name in the filename, or reuse one directory for multiple generation runs.
Separate model/run directories prevent accidental overwrites and make results
reproducible.

For arbitrary generation filenames, use a JSONL video manifest instead:

```json
{"sample_id":"0001","video_path":"videos/custom_name.mp4"}
```

Manifest paths are resolved relative to the manifest file.

## 3. Evaluator Result Directory

Use one output directory per evaluator run:

```text
evaluation/
└── <model_name>/
    └── <generation_run_id>/
        └── ref2v/
            ├── summary.json
            ├── run_details.json
            ├── 0001/
            │   └── result.json
            └── 1190/
                └── result.json
```

The formal four-digit sample ID is used directly as the result directory name.
The evaluator still applies defensive filename sanitization to tolerate
non-conforming programmatic inputs. The original `sample_id` remains inside
`result.json` and `summary.json`.

Normal output contains final metrics and sample metadata. It also writes
`run_details.json` for execution diagnostics, but does not embed per-frame OCR
or VLM traces in the public metric files.

`summary.json` contains the batch counts, successful result paths, complete
final metrics copied from each successful sample's `result.json`, per-sample
errors, update time, and a benchmark-level arithmetic mean over successful
samples:

```json
{
  "total": 1,
  "ok": 1,
  "error": 0,
  "results": [
    {
      "sample_id":"0001",
      "status":"ok",
      "result":"0001/result.json",
      "metrics": {
        "length_weighted_spatial_fidelity": 0.91,
        "length_weighted_temporal_fidelity": 0.88,
        "textv_score": 89.49,
        "extra_text_ratio": 0.04,
        "reading_order_score": 0.95
      }
    }
  ],
  "errors": [],
  "benchmark_metrics": {
    "Spatial": 0.91,
    "Temporal": 0.88,
    "TextVScore": 89.49,
    "Extra": 0.04,
    "spatial_fidelity": 0.91,
    "temporal_fidelity": 0.88,
    "textv_score": 89.49,
    "extra_text_ratio": 0.04,
    "sample_count": 1,
    "aggregation": "arithmetic_mean_over_successful_samples"
  },
  "updated_at": "..."
}
```

`benchmark_metrics.Spatial`, `Temporal`, `TextVScore`, and `Extra` are the
publication-facing aliases. Spatial, Temporal, and Extra retain their native
`[0, 1]` scale; TextVScore is on the `[0, 100]` scale. The same object is
written by the T2V evaluator and its parallel merge step.

The `metrics` object in a successful `results[]` entry is the complete
`result.json.metrics` object, including per-target and per-reference values;
the example shows only the most commonly inspected fields. Failed samples do
not receive fabricated metrics.

`run_details.json` stores execution diagnostics separately from the compact
metric summary. It records the effective input/output locations, video source
arguments, sample filter, debug directory, run timing, per-sample elapsed time,
resolved video path, and exception text. It is intended for troubleshooting
failed or slow runs and is not part of the benchmark metric contract.

Missing or unreadable video files are recorded as ordinary per-sample errors so
the remaining samples can continue. PaddleOCR initialization/inference failures
and VLM initialization/connection/request failures are dependency-level fatal
errors and stop the evaluator; malformed VLM JSON or a business-validation
failure remains a retryable sample error.

The evaluator is resumable when the same command and output directory are
used. A valid existing `<sample_id>/result.json` is treated as a successful
checkpoint and skipped; samples with no valid result, including previous
`error` samples, are evaluated again. Use `--override` to evaluate every
selected sample again. Successful checkpoints are written immediately after
each sample, so an interrupted process can continue from the next sample.

## 4. Debug Directory

Debug artifacts should be enabled separately and kept out of the public
metrics tree:

```text
evaluation/
└── <model_name>/<generation_run_id>/
    ├── ref2v/
    │   ├── summary.json
    │   ├── run_details.json
    │   └── <sample_id>/result.json
    └── ref2v_debug/
        └── <sample_id>/
            ├── input_record.json
            ├── resolved_config.json
            ├── manifest.json
            ├── prompts/
            │   ├── ref_id_matching_system_prompt.md
            │   └── target_candidate_matching_system_prompt.md
            ├── reference_R01.jpg
            ├── reference_R01_ocr.json
            ├── frames/
            │   └── t_<timestamp>/
            │       ├── source_frame.jpg
            │       ├── marker.jpg
            │       ├── ocr_and_filtering.json
            │       ├── ref_id_matching_vlm_call.json       # multi-ref only
            │       ├── matching_marker_batch_000.jpg
            │       ├── matching_vlm_call_batch_*.json
            │       └── observation.json
            └── result.json
```

The debug `result.json` includes frame observations; the public result does
not. Debug artifacts may contain model prompts and candidate/matching details
and should not be treated as the compact benchmark result.

`reference_Rxx.jpg` is the exact compact reference input used for VLM
grounding. Its upper-left label identifies the corresponding `ref_id`; `Rxx`
reference IDs must not be confused with `Pxxx` candidate IDs drawn on marker
images.

## 5. Invocation

From the TextVBench repository root:

```bash
PYTHONPATH=src python3 -m textvbench evaluate ref2v \
  --input data/Ref2V \
  --video-dir /path/to/generated/<model_name>/<generation_run_id>/videos \
  --output /path/to/evaluation/<model_name>/<generation_run_id>/ref2v \
  --debug-dir /path/to/evaluation/<model_name>/<generation_run_id>/ref2v_debug \
  --config configs/ref2v.yaml \
  --sample-workers 4 \
  --project-root .
```

Reusing the same command and output directory resumes automatically: samples
with a valid existing `result.json` are skipped, while missing and previous
`error` samples are evaluated again. Add `--override` to force evaluation of
all selected samples, including successful checkpoints. A successful result
is written after each sample, so an interrupted batch can continue on the next
run.

`--sample-workers` controls bounded sample-level concurrency and defaults to
`1` (or `runtime.sample_workers` in the YAML file). Only different samples
run concurrently. A sample's frame loop, Ref-ID routing, target batches, and
candidate-removal state remain serial, so the matching semantics are unchanged.
The shared PaddleOCR instance is protected by a lock for both first-use model
construction and inference; increasing this value therefore primarily overlaps
video work and VLM requests across samples. Start with `2` or `4` and verify
the VLM service's capacity before using a higher value.

For a one-sample smoke test, pass `--sample-id` and `--video`; `--video`
requires exactly one selected sample. Do not pass both `--video` and
`--video-dir`/`--video-manifest`.

The YAML file is the source of runtime model settings. Before a real
evaluation, set the OCR model directories and the active OpenAI-compatible VLM
profile in that file. `matching.target_batch_size` controls the maximum number
of slots sent in one matching request (the default is 3). The evaluator also
prints sample-level and frame-level progress bars while it runs.

## 6. Read-Only Evaluator Review

### 6.1 Required artifacts

The Review UI explains an existing evaluation; it does not rerun the evaluator.
Run `evaluate` with `--debug-dir` first. The viewer then consumes:

```text
--data / REVIEW_DIR          required debug root, or one sample debug directory
--results-dir / RESULTS_DIR  optional public result root from the same run
--video-dir / VIDEO_DIR      optional canonical <sample_id> video directory
--video-manifest             optional mapping for arbitrary video filenames
```

When `RESULTS_DIR` is present, its `<sample_id>/result.json` is the source of
official metrics. The debug result remains the source of frame-level evidence.
Without `RESULTS_DIR`, the viewer falls back to the debug result for both.

### 6.2 Launcher

From the repository root:

```bash
REVIEW_DIR=/path/to/evaluation/<model>/<run>/ref2v_debug \
RESULTS_DIR=/path/to/evaluation/<model>/<run>/ref2v \
VIDEO_DIR=/path/to/generated/<model>/<run>/videos \
./scripts/review/start_ref2v.sh
```

Open `http://127.0.0.1:8768`.

The launcher supports these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `REVIEW_DIR` | required | Debug root or one sample debug directory |
| `RESULTS_DIR` | unset | Public evaluator output root |
| `VIDEO_DIR` | unset | Videos named `<sample_id>.mp4`, `.mov`, `.webm`, or `.avi` |
| `VIDEO_MANIFEST` | unset | JSONL mapping arbitrary video names to sample IDs |
| `HOST` | `127.0.0.1` | Listener address |
| `PORT` | `8768` | Listener port |
| `PYTHON_BIN` | `python3` | Python executable containing project dependencies |
| `FRONTEND_DIR` | `frontend/ref2v_review` | Static Review frontend |

Relative paths are resolved from the Ref2V repository root. For remote
development environments, use `HOST=0.0.0.0` and forward the configured port;
the default remains local-only.

The equivalent module invocation is:

```bash
PYTHONPATH=src python3 -m textvbench.evaluators.ref2v.review_api.server \
  --data /path/to/ref2v_debug \
  --results-dir /path/to/ref2v \
  --video-dir /path/to/videos \
  --frontend frontend/ref2v_review \
  --host 127.0.0.1 \
  --port 8768
```

After package installation, `textvbench-ref2v-review` exposes the same CLI.

### 6.3 Non-canonical video filenames

`VIDEO_DIR` discovers canonical filenames by sample ID. For arbitrary names,
use a JSONL manifest:

```json
{"sample_id":"1190","video_path":"videos/custom-generation-name.mp4"}
```

Relative `video_path` values are resolved from the manifest's directory.
Absolute paths are accepted as local server configuration but are never
returned by the API. If both video sources are configured, a manifest entry
takes precedence and `VIDEO_DIR` acts as fallback for unmapped samples.

### 6.4 Review behavior

The case list provides sample-ID search, status filtering, pagination, and
sorting by TextVScore, spatial fidelity, temporal fidelity, ExtraTextRatio,
reading order, or sample ID. The detail view exposes:

- final global, per-reference, and per-target metrics;
- labeled reference images and runtime OCR slots;
- source video and sampled-frame images;
- active, filtered, duplicate-extra, and capped-extra OCR candidates;
- Ref-ID routing for multi-reference samples, with an explicit skipped state
  for single-reference samples;
- target-candidate batches, including one candidate reused by multiple targets
  in the same batch;
- VLM system prompts, payloads, responses, attempts, retry failures, and final
  errors;
- per-frame reading-order evidence and per-target temporal observations;
- sanitized manifest, input record, config, result, and run details.

No-match frames are shown as excluded, consistent with metric aggregation. A
failed VLM attempt followed by a successful retry increments the retry count
but does not mark the case as failed. Chinese and English UI labels are
available from the header switch.

### 6.5 Security boundary

The Review service is read-only. It has no result-editing or evaluator-control
endpoint. API routes only serve known JSON/JPEG artifacts under the selected
case, while videos must be explicitly configured. Credentials, private VLM
endpoints, and machine-specific absolute paths are removed from returned JSON.
Videos support HTTP Range requests for browser seeking.

The raw debug directory can still contain full prompts, endpoints, filesystem
paths, generated images, and model responses. Do not publish it directly as a
static directory. When exposing the Review service outside localhost, place it
behind an authenticated reverse proxy and review the benchmark content itself
for sensitive text or imagery.
