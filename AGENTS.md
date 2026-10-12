# Instructions for Agents Using TextVBench

This file is the operational guide for AI coding agents, research agents, and
automation systems that are asked to use TextVBench. Assume the user may know
nothing about this repository. Translate their goal into the correct benchmark
track, prepare the inputs, run a bounded smoke test, run the requested
evaluation, validate coverage, and report the metrics with their limitations.

All commands in this file run from the repository root. Tokens such as
`<model>`, `<run>`, and `<sample_id>` are placeholders. Replace them before
executing a command; literal angle brackets have shell syntax and must never be
passed unchanged.

## 1. What TextVBench Does

TextVBench measures visual-text fidelity in generated video. It contains two
tracks with different ground-truth sources:

| User need | Track | Ground truth | Public input |
| --- | --- | --- | --- |
| Evaluate text requested by a text prompt | T2V | Explicit carrier, slot, and bilingual target text | `data/T2V/benchmark_300.jsonl` |
| Evaluate text preservation from reference images | Ref2V | OCR text and reading order extracted from the reference images at evaluation time | `data/Ref2V/` |

Use T2V when the generation input is a text prompt and the requirement is to
render specified text in the scene. Use Ref2V when generation receives one or
more reference images and should preserve their visible text. If a system
supports both modes, evaluate each mode as a separate run. Never merge T2V and
Ref2V results into one score without an explicitly defined external protocol.

TextVBench does not:

- generate videos;
- launch or configure a video-generation model;
- evaluate general aesthetics, motion quality, prompt alignment, identity, or
  safety outside visual-text fidelity;
- provide an official model-level leaderboard aggregation script;
- require or expose the dataset-construction pipelines used before release.

If the user's main need is outside these boundaries, explain that limitation
before running the benchmark. Do not present TextVScore as a general video
quality score.

If generated videos are not available, evaluation cannot start. Do not use the
reference images, source videos, or unrelated media as substitutes. An agent
may prepare the generation inputs and expected output mapping, but integrating
or running a generation model is a separate task that requires an available
model interface and the user's authority.

When the request is only to run an evaluation, do not modify evaluator code,
system prompts, schemas, release configs, or public data. Prefer environment
variables and run-specific config copies. Modify implementation files only
when the user explicitly asks to develop or fix the benchmark.

## 2. Repository Sources of Truth

Use these files in this order:

1. `AGENTS.md`: workflow and reporting rules for agents.
2. `README.md`: installation and public CLI examples.
3. `DATASET.md`: public dataset contract.
4. `configs/t2v.yaml` and `configs/ref2v.yaml`: evaluator defaults.
5. `docs/t2v_evaluator.md`, `docs/ref2v_evaluator.md`, and
   `docs/ref2v_io.md`: metric and artifact details.
6. `src/textvbench/evaluators/`: executable behavior when documentation and
   code disagree.

The benchmark JSONL and reference images are immutable evaluation inputs. Do
not rewrite records, rename reference images, insert generated-video paths into
the dataset, or publish debug output as benchmark data.

`data/Ref2V/sampling_report.json` documents release sampling provenance. It is
not an evaluator input and must not be passed to the CLI.

## 3. Required Decisions Before Running

Resolve the following from the user's request or the available files:

- track: `t2v` or `ref2v`;
- model name and generation run ID;
- generated-video directory or video manifest;
- T2V language: `en` or `zh`;
- OpenAI-compatible multimodal VLM endpoint and model ID;
- PaddleOCR device and optional local model directories;
- whether metrics-only output is sufficient or debug evidence is required;
- whether this is a smoke test, partial evaluation, or full benchmark run.

T2V language is never inferred. English and Chinese videos must be generated
and evaluated in separate runs. A T2V video generated from `prompt_en` must be
evaluated with `--language en`; a video generated from `prompt_zh` must use
`--language zh`.

Use a unique output directory for every combination of:

```text
track / model / generation-run / language / evaluator-config
```

This is a correctness requirement. Both evaluators resume by skipping existing
successful sample IDs. Reusing an output directory after changing the video,
language, prompt, config, evaluator prompt, OCR model, or VLM can silently mix
results. Use a new directory whenever possible; otherwise pass `--override`.

## 4. Environment Preflight

Use Python 3.10 or newer. A typical editable installation is:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,ocr]'
```

PaddleOCR does not install the correct PaddlePaddle runtime for every CPU/CUDA
environment. Install a compatible PaddlePaddle build separately, then verify
that both `paddle` and `paddleocr` import successfully.

Before evaluating, verify:

```bash
.venv/bin/python -c "import cv2, openai, paddle, paddleocr, PIL, yaml"
.venv/bin/textvbench --help
.venv/bin/python -m pytest -q
```

The evaluator VLM must expose an OpenAI-compatible chat-completions endpoint,
accept image data URLs, and return JSON-object responses. Configure it without
committing credentials:

```bash
export TEXTVBENCH_VLM_BASE_URL=http://127.0.0.1:8000/v1
export TEXTVBENCH_VLM_MODEL=your-vision-language-model
export TEXTVBENCH_VLM_API_KEY=EMPTY
```

`TEXTVBENCH_VLM_PROFILE` optionally selects a profile name. Environment values
override the selected profile in both evaluator YAML files.

Relative prompt, font, and OCR model paths resolve from `--project-root`.
Run from the repository root and keep the default `--project-root .` unless
there is a concrete reason to change it.

Default OCR execution is on CPU. Do not change it to GPU without checking free
device memory and installing a matching PaddlePaddle GPU build. Empty OCR model
paths allow PaddleOCR to use its managed models; in an offline environment,
set `detection_model_dir` and `recognition_model_dir` to existing local paths.

## 5. Prepare Generated Videos

### 5.1 T2V generation contract

For every selected JSONL record:

1. Preserve `sample_id` exactly.
2. Generate from `prompt_en` for an English run or `prompt_zh` for a Chinese
   run. Do not translate or rewrite the public prompt for an official run.
3. Save the video as `<sample_id>.mp4` in one run-specific directory, or create
   a video manifest.
4. Do not combine English and Chinese outputs in the same evaluation run.

The `carriers` and `texts` fields are evaluator ground truth. They must not be
sent to the generation model as hidden extra instructions unless the evaluated
system's declared interface explicitly includes them. The public generation
input is the selected prompt field.

### 5.2 Ref2V generation contract

For every selected JSONL record:

1. Preserve `sample_id` exactly.
2. Use `prompt` as the textual generation instruction.
3. Supply every entry in `reference_images` from `data/Ref2V/imgs/`, preserving
   the listed order and `ref_id` association.
4. Save the video as `<sample_id>.mp4` in one run-specific directory, or create
   a video manifest.

Do not preprocess the public reference images for evaluation. The evaluator
runs OCR on the original files. Generation-time preprocessing required by a
model is allowed, but record it as part of the model protocol.

### 5.3 Video mapping modes

Use exactly one mapping mode:

- `--video-dir`: canonical batch mode. Supported suffixes are `.mp4`, `.mov`,
  `.webm`, and `.avi`.
- `--video-manifest`: JSONL mapping for arbitrary filenames. Each row must
  contain unique `sample_id` and `video_path` values. Relative video paths are
  resolved from the manifest's directory.
- `--video` with `--sample-id`: one-sample smoke test only.

Example manifest:

```json
{"sample_id":"0001","video_path":"videos/custom-name.mp4"}
```

Do not infer sample-to-video mapping from directory order, file timestamps, or
similar filenames.

## 6. Mandatory Smoke Test

Never begin an expensive full run before one real sample succeeds end to end.
Choose a sample ID that exists in the relevant public JSONL and a corresponding
generated video.

T2V smoke test:

```bash
.venv/bin/textvbench evaluate t2v \
  --input data/T2V/benchmark_300.jsonl \
  --sample-id <sample_id> \
  --video /absolute/path/to/<sample_id>.mp4 \
  --language en \
  --max-frames 1 \
  --output evaluation/<model>/<run>/t2v-en-smoke/metrics.jsonl \
  --debug-dir evaluation/<model>/<run>/t2v-en-smoke/debug
```

Ref2V smoke test:

```bash
.venv/bin/textvbench evaluate ref2v \
  --input data/Ref2V \
  --sample-id <sample_id> \
  --video /absolute/path/to/<sample_id>.mp4 \
  --output evaluation/<model>/<run>/ref2v-smoke/metrics \
  --debug-dir evaluation/<model>/<run>/ref2v-smoke/debug
```

Before scaling up, confirm all of the following:

- the video opens and has a nonzero frame count and duration;
- PaddleOCR initializes and returns a valid result, including an empty result
  when no text is visible;
- the configured VLM accepts the marker image and returns valid JSON;
- the sample ID, language, prompt, reference images, and video correspond;
- the result contains `metrics.textv_score`;
- the debug bundle contains frames and OCR evidence, plus VLM traces for every
  judgment that invoked the VLM. A valid programmatic fast-match can require no
  target-matching VLM call.

`--max-frames` is currently a T2V CLI option. To bound a Ref2V smoke test,
create a temporary copy of `configs/ref2v.yaml` and reduce
`sampling.max_frames`; do not change the release config for one experiment.

## 7. Full Evaluation Commands

T2V English example:

```bash
.venv/bin/textvbench evaluate t2v \
  --input data/T2V/benchmark_300.jsonl \
  --video-dir generated/<model>/<run>/t2v-en/videos \
  --language en \
  --output evaluation/<model>/<run>/t2v-en/metrics.jsonl
```

Run Chinese videos separately with `--language zh` and a different video and
output directory.

Ref2V example:

```bash
.venv/bin/textvbench evaluate ref2v \
  --input data/Ref2V \
  --video-dir generated/<model>/<run>/ref2v/videos \
  --output evaluation/<model>/<run>/ref2v/metrics \
  --sample-workers 4
```

Start Ref2V with `--sample-workers 1`, then increase to `2` or `4` only after
checking VLM capacity. PaddleOCR is shared and serialized across Ref2V workers;
additional workers primarily overlap video decoding and VLM requests. More
workers do not guarantee proportional speedup.

Debug artifacts are useful but large. Enable `--debug-dir` for smoke tests,
failed cases, or audit runs. Metrics-only mode is preferable for routine full
runs when frame-level evidence is not required.

## 8. Resume and Failure Semantics

Both tracks resume successful samples when the same output location is reused.
Failed or missing samples are retried. `--override` forces reevaluation of all
selected samples.

Errors have two classes:

- Sample errors, such as a missing video or malformed record, are recorded for
  that sample and the batch can continue.
- Dependency errors from PaddleOCR or the VLM are fatal because continuing
  would make later scores invalid or incomparable.

The CLI exits nonzero when a completed batch contains sample errors. A nonzero
exit is therefore a signal to inspect the result summary, not permission to
discard the output. A fatal dependency error may stop before a complete summary
is written.

After every run, verify `total`, `ok`, and `error`. Do not calculate or report a
model score until `error == 0`, unless the user explicitly requests a partial
diagnostic result. Partial results must be labeled as partial and include the
exact successful/expected count.

A full T2V run contains 300 samples per language. A full Ref2V run contains 200
samples. A user-requested subset is valid, but it must be labeled with its
selection rule and sample count rather than described as a full benchmark run.

Changing any evaluation input or setting invalidates resume compatibility.
This includes VLM endpoint/model, OCR model, thresholds, sampling settings,
prompts, generated video, language, or evaluator code.

## 9. Output Layout

### 9.1 T2V

Given:

```text
--output evaluation/<model>/<run>/t2v-en/metrics.jsonl
```

the evaluator writes:

```text
evaluation/<model>/<run>/t2v-en/
├── passed/metrics.jsonl
├── failed/metrics_errors.jsonl
└── summary.json
```

Each successful row retains the benchmark record and adds:

```text
evaluator_t2v.audit
evaluator_t2v.metrics
```

The top-level `summary.json` also contains `benchmark_metrics`, an arithmetic
mean over successful rows with publication-facing `Spatial`, `Temporal`,
`TextVScore`, and `Extra` fields (and snake_case aliases).

With `--debug-dir`, each case directory is named
`<sample_id>_<language>/`.

### 9.2 Ref2V

Given:

```text
--output evaluation/<model>/<run>/ref2v/metrics
```

the evaluator writes:

```text
evaluation/<model>/<run>/ref2v/metrics/
├── summary.json
├── run_details.json
└── <sample_id>/result.json
```

`summary.json` includes per-sample metrics for successful records and explicit
errors for failed records. `run_details.json` contains execution diagnostics
and local paths; treat it as a local artifact rather than publication-ready
benchmark data. The summary also contains a top-level `benchmark_metrics`
object with arithmetic means over successful samples: `Spatial`, `Temporal`,
`TextVScore`, and `Extra`, plus their snake_case aliases and the sample count.

## 10. Metric Interpretation

All fidelity values except TextVScore are in `[0, 1]`. TextVScore is in
`[0, 100]`.

| Metric | Direction | Meaning |
| --- | --- | --- |
| `spatial_fidelity` | higher is better | Target-text similarity on the intended visible carrier across evaluated frames |
| `temporal_fidelity` | higher is better | Consistency of matched text between adjacent evaluated observations |
| `textv_score` | higher is better | `100 *` harmonic mean of spatial and temporal fidelity |
| `extra_text_ratio` | lower is better | OCR text not assigned to requested targets relative to evaluated text |
| `reading_order_score` | higher is better | Ref2V-only preservation of reference reading order, including missing-slot penalties |

Ref2V also exposes `length_weighted_spatial_fidelity`,
`length_weighted_temporal_fidelity`, per-target metrics, and per-reference
metrics. Its short `spatial_fidelity` and `temporal_fidelity` aliases carry the
same video-level values.

Important interpretation rules:

- `extra_text_ratio` is diagnostic and is not part of TextVScore.
- A high TextVScore does not imply low extra text; report both.
- TextVScore is a harmonic mean, not an arithmetic average.
- Longer target strings receive more weight in video-level spatial and temporal
  fidelity.
- Ref2V target text comes from runtime OCR on the reference image. OCR quality
  is therefore part of the operational evaluation pipeline.
- T2V carrier absence and Ref2V unmatched references can produce zero credit;
  inspect debug evidence before attributing the failure to text spelling alone.

## 11. Model-Level Reporting

The repository produces per-sample metrics, coverage information, and an
explicitly labeled arithmetic-mean `benchmark_metrics` object in each
summary. This is a reporting convenience, not an official cross-sample
leaderboard score. Do not invent an "official TextVBench score."

When the user requests an exploratory model summary, use an explicitly labeled
aggregation, normally the arithmetic mean over the complete successful sample
set, and state the aggregation rule. Always report:

- repository version or commit;
- benchmark track and T2V language;
- model and generation run;
- evaluator VLM model;
- OCR device/model configuration;
- sampling and threshold configuration;
- expected, successful, and failed sample counts;
- mean TextVScore, spatial fidelity, temporal fidelity, and ExtraTextRatio;
- Ref2V reading-order score when applicable;
- T2V difficulty-sliced results for `easy`, `medium`, and `hard` when reporting
  a full T2V run.

Do not average English and Chinese T2V runs without also reporting them
separately. Do not compare two models evaluated with different evaluator VLMs,
OCR models, configs, prompts, sampling settings, or dataset revisions as if the
scores were directly comparable.

## 12. Review Debug Evidence

The included frontends are read-only viewers. They explain existing debug
bundles and do not rerun the evaluator.

T2V:

```bash
REVIEW_DIR=./evaluation/<model>/<run>/t2v-en/debug \
VIDEO_DIR=./generated/<model>/<run>/t2v-en/videos \
./scripts/review/start_t2v.sh
```

Open `http://127.0.0.1:8767`.

Ref2V:

```bash
REVIEW_DIR=./evaluation/<model>/<run>/ref2v/debug \
RESULTS_DIR=./evaluation/<model>/<run>/ref2v/metrics \
VIDEO_DIR=./generated/<model>/<run>/ref2v/videos \
./scripts/review/start_ref2v.sh
```

Open `http://127.0.0.1:8768`.

Use `HOST=0.0.0.0` only when port forwarding is required. Raw debug bundles
may contain prompts, images, generated content, endpoints, credentials copied
from configuration, and machine-specific paths. Never publish a raw debug
directory or expose a review server publicly without authentication and a
content/privacy review.

## 13. Troubleshooting Order

Diagnose failures in this order:

1. **Wrong track or language**: confirm T2V vs Ref2V and the T2V prompt variant
   used for generation.
2. **Sample mapping**: confirm exact sample IDs, canonical filenames, manifest
   uniqueness, and path resolution.
3. **Video decoding**: verify OpenCV can open the file and reports nonzero FPS
   and frame count.
4. **Reference input**: for Ref2V, confirm every listed image exists under the
   inferred or explicit image directory.
5. **OCR runtime**: check Paddle/PaddleOCR imports, device compatibility, model
   paths, cache permissions, and memory.
6. **VLM runtime**: check endpoint reachability, model ID, API key, multimodal
   support, JSON response support, timeout, and server capacity.
7. **Evidence quality**: inspect OCR candidates, marker images, carrier/ref-ID
   routing, matching responses, and retries in the debug UI.
8. **Resume contamination**: verify that the output directory was not reused
   across languages, models, videos, or configs.

Common symptoms:

- `--video requires --sample-id`: add the exact sample filter or use
  `--video-dir`/`--video-manifest`.
- `no matching Ref2V samples`: the sample ID is absent or was parsed with the
  wrong formatting; Ref2V IDs are four-character strings such as `"0001"`.
- missing video errors: use canonical `<sample_id>.<suffix>` filenames or a
  manifest; do not rely on embedded dataset paths.
- PaddleOCR import or initialization failure: install a compatible PaddlePaddle
  runtime and validate the configured device.
- VLM request failure: treat it as fatal; do not replace missing judgments with
  zeros and continue.
- unexpectedly skipped samples: the output already contains successful IDs;
  choose a fresh directory or use `--override` after confirming the target.
- low score with apparently readable text: check carrier/ref association,
  language, OCR output, split OCR boxes, extra text, and temporal instability.
- review UI shows no cases: confirm `--debug-dir` was used and point
  `REVIEW_DIR` at the debug root, not the metrics directory.

## 14. Rules for Comparable and Reproducible Runs

For any result intended for comparison or publication:

- keep public benchmark files unchanged;
- use the same evaluator commit and dependency versions;
- use identical evaluator configs, system prompts, OCR models, and evaluator
  VLM for all compared generation models;
- retain the video manifest or canonical video directory listing;
- retain sanitized config and run metadata, excluding API keys;
- record failures rather than dropping difficult samples;
- use a fresh output directory for each run;
- disclose any subset, changed threshold, reduced frame limit, or custom
  aggregation;
- never hand-edit evaluator output to repair a score.

If evaluator code, prompts, schemas, or default configs are modified, run the
full test suite before evaluation:

```bash
.venv/bin/python -m pytest -q
```

Code changes create a new evaluator version. Do not compare new results with
old cached results without rerunning both sides.

## 15. Completion Checklist for an Agent

An evaluation task is complete only when the agent has:

- selected and stated the correct track;
- stated T2V language when applicable;
- verified environment, OCR, VLM, and input mappings;
- completed a real one-sample smoke test;
- run the requested sample set;
- checked expected vs successful vs failed counts;
- inspected debug evidence for anomalies or representative failures;
- reported metrics with units, directions, coverage, and aggregation method;
- disclosed deviations from the default config or full public benchmark;
- preserved reproducibility metadata without exposing secrets.

If any required input or authority is missing, stop before fabricating data or
changing the benchmark contract. Report exactly what is missing and provide the
next concrete command or file the user must supply.

Use this minimum handoff shape after a run:

```text
Track: T2V (en) | T2V (zh) | Ref2V
Generation model/run: ...
Evaluator version and VLM: ...
Coverage: <ok>/<expected>, <error> errors
Configuration deviations: none | ...
Metrics and direction: ...
Slices: difficulty or other declared subset breakdowns
Failures/anomalies: none | ...
Artifacts: metrics, summary, debug, and manifest paths
Aggregation: per-sample only | explicitly named model-level aggregation
```
