# Dataset

## T2V

`data/T2V/benchmark_300.jsonl` contains 300 bilingual prompts, balanced across
`easy`, `medium`, and `hard` difficulty levels. Each line contains:

- `sample_id`: stable identifier used for generated-video filenames;
- `difficulty`: benchmark difficulty slice;
- `carriers`: requested visual carriers and their text slots;
- `texts`: English and Chinese target text for every slot;
- `prompt_en` and `prompt_zh`: parallel generation prompts.

Generate English and Chinese videos as separate runs. The evaluator requires
`--language en` or `--language zh` and never infers the language from a video.

## Ref2V

`data/Ref2V/video_prompt_dataset.jsonl` contains 200 samples and
`data/Ref2V/imgs/` contains the corresponding 200 reference images. Each JSONL
line contains:

- `sample_id`: four-digit stable identifier;
- `prompt`: the generation instruction;
- `reference_images`: one or more `{ref_id, filename}` objects.

Reference paths are relative to `data/Ref2V/imgs/`. The public records omit
source paths, categories, precomputed OCR, and construction metadata. The
evaluator discovers all visible reference text with OCR at runtime.

`data/Ref2V/sampling_report.json` records the deterministic seed, source
allocation, and selected distribution used to create the release subset. It is
provenance documentation and is not an evaluator input.

## Generated videos

Generated videos are not included. The canonical batch layout is:

```text
generated/<model>/<run>/videos/<sample_id>.mp4
```

Supported extensions are `.mp4`, `.mov`, `.webm`, and `.avi`. A video manifest
can map non-canonical filenames:

```json
{"sample_id":"0001","video_path":"videos/custom-name.mp4"}
```

Relative manifest paths resolve from the manifest file's directory.
