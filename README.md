# TextVBench

[Project Page](https://zhang-zrq.github.io/textvbench-page/) | [Paper](https://arxiv.org/abs/2610.13412)

TextVBench evaluates visual-text rendering in generated videos. This release
contains two complementary benchmarks:

- **T2V** tests whether a text-to-video model renders requested text on the
  intended carrier and keeps it stable over time.
- **Ref2V** tests whether a reference-to-video model preserves the visible
  text and reading order found in one or more reference images.

The repository includes the public benchmark data, both evaluators, evaluator
prompts and schemas, tests, and read-only review interfaces for inspecting
debug evidence. Dataset-construction pipelines and intermediate artifacts are
intentionally excluded.

## Repository layout

```text
TextVBench/
├── data/
│   ├── T2V/benchmark_300.jsonl
│   └── Ref2V/
│       ├── video_prompt_dataset.jsonl
│       ├── sampling_report.json
│       └── imgs/
├── src/textvbench/evaluators/{t2v,ref2v}/
├── configs/{t2v,ref2v}.yaml
├── prompts/{t2v,ref2v}/
├── schemas/
├── frontend/{t2v_review,ref2v_review}/
├── scripts/{review,vllm}/
└── tests/
```

Agents and automated evaluation systems should start with
[AGENTS.md](AGENTS.md). See [DATASET.md](DATASET.md) for the public data
contract. Metric details are documented in
[docs/t2v_evaluator.md](docs/t2v_evaluator.md) and
[docs/ref2v_evaluator.md](docs/ref2v_evaluator.md).

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,ocr]'
```

Install a CPU or GPU build of PaddlePaddle appropriate for the host before the
`ocr` extra. The correct PaddlePaddle command depends on the platform and CUDA
version; follow the official PaddleOCR installation instructions. The base
package includes OpenCV, Pillow, and the OpenAI-compatible VLM client.

### Run a local VLM

The repository includes a small vLLM wrapper for Qwen3.8-27B. Install vLLM
and the Hugging Face CLI in an environment suitable for your GPU, then run:

```bash
./scripts/vllm/download_model.sh

VLLM_BIN=/path/to/vllm \
./scripts/vllm/run_evaluator.sh \
  .venv/bin/textvbench evaluate t2v \
  --input data/T2V/benchmark_300.jsonl \
  --video-dir /path/to/generated/t2v/videos \
  --language en \
  --output evaluation/model/run/t2v/metrics.jsonl
```

`run_evaluator.sh` starts vLLM, waits for `/health`, configures the evaluator,
and stops the server when evaluation exits. The defaults target one 80 GB GPU.
Use `MODEL_DIR` while downloading, and `MODEL_PATH`, `VLLM_BIN`, `VLLM_PORT`,
`VLLM_TP_SIZE`, or `VLLM_GPU_MEMORY_UTILIZATION` while running to override
them. The server log is written to `output/vllm/server.log`.

Configure the vision-language model in `configs/t2v.yaml` and
`configs/ref2v.yaml`, or leave the files unchanged and use environment
variables:

```bash
export TEXTVBENCH_VLM_BASE_URL=http://127.0.0.1:8000/v1
export TEXTVBENCH_VLM_MODEL=your-vision-language-model
export TEXTVBENCH_VLM_API_KEY=EMPTY
```

OCR model directories are optional. Empty paths let PaddleOCR manage its model
files; local paths can be set in each YAML configuration.

## Evaluate T2V

Generate one video per sample and name it `<sample_id>.mp4`. Evaluate English
and Chinese generations in separate runs because the language is an explicit
part of the ground-truth contract.

```bash
textvbench evaluate t2v \
  --input data/T2V/benchmark_300.jsonl \
  --video-dir /path/to/generated/t2v/videos \
  --language en \
  --output evaluation/model/run/t2v/metrics.jsonl
```

The evaluator writes successful records to `passed/metrics.jsonl`, failures to
`failed/metrics_errors.jsonl`, and aggregate counts to `summary.json`. Add
`--debug-dir evaluation/model/run/t2v_debug` to retain sampled frames, OCR
candidates, marker images, and VLM traces for review. For a single
smoke test, combine `--sample-id` with `--video` and optionally
`--max-frames 1`.

## Evaluate Ref2V

Ref2V resolves reference filenames under `data/Ref2V/imgs` and runs OCR on the
references at evaluation time.

```bash
textvbench evaluate ref2v \
  --input data/Ref2V \
  --video-dir /path/to/generated/ref2v/videos \
  --output evaluation/model/run/ref2v \
  --debug-dir evaluation/model/run/ref2v_debug \
  --sample-workers 4
```

Ref2V writes `summary.json`, `run_details.json`, and one
`<sample_id>/result.json` per successful sample. Both evaluators resume from
successful results by default; use `--override` for a full rerun. If generated
filenames do not match sample IDs, pass a JSONL `--video-manifest` containing
`sample_id` and `video_path`.

## Review debug evidence

The review servers do not rerun evaluation or modify evaluator artifacts.

```bash
REVIEW_DIR=./evaluation/model/run/t2v_debug \
VIDEO_DIR=./generated/t2v/videos \
./scripts/review/start_t2v.sh

REVIEW_DIR=./evaluation/model/run/ref2v_debug \
RESULTS_DIR=./evaluation/model/run/ref2v \
VIDEO_DIR=./generated/ref2v/videos \
./scripts/review/start_ref2v.sh
```

Open `http://127.0.0.1:8767` for T2V or `http://127.0.0.1:8768` for Ref2V.
At startup, each server scans its `REVIEW_DIR`, replaces its cache file
`.textvbench_review_index.json` atomically, and keeps that index in memory for
fast filtering and pagination. The Refresh button explicitly rebuilds the same
index so newly written evaluator artifacts become visible without restarting.
Review status is sample-level: a case is `complete` when its result contains
metrics, even if an individual VLM call retried or failed; otherwise it is
`error`. Call-level failures remain visible in the case evidence.
Raw debug directories can contain prompts, generated imagery, endpoints, and
machine paths. Do not publish them directly.

## Citation

If you use TextVBench in your research, please cite our paper:

```bibtex
@misc{zhang2026textvbenchevaluatingtextrendering,
      title={TextVBench: Evaluating Text Rendering and Preservation in Video Generation},
      author={Ruiqiang Zhang and Hengyi Wang and Chang Liu and Guanjie Wang and Zehua Ma and Weiming Zhang},
      year={2026},
      eprint={2610.13412},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.13412},
}
```

## License

Code and documentation are available under the [Apache License 2.0](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for reference imagery,
external dependencies, and model terms.
