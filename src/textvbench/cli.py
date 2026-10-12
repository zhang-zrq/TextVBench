"""Unified command line interface for TextVBench evaluators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .evaluators.ref2v.pipeline import run as run_ref2v
from .evaluators.ref2v.vlm import VLMClient as Ref2VVLMClient
from .evaluators.t2v.pipeline import run as run_t2v


def _add_video_arguments(parser: argparse.ArgumentParser) -> None:
    videos = parser.add_mutually_exclusive_group()
    videos.add_argument("--video", help="video path for a single --sample-id")
    videos.add_argument("--video-dir", help="directory containing <sample_id> video files")
    videos.add_argument("--video-manifest", help="JSONL mapping sample_id to video_path")
    parser.add_argument("--sample-id", help="evaluate one sample")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="textvbench")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "evaluator"):
        evaluate = commands.add_parser(command, help="run a benchmark evaluator")
        benchmarks = evaluate.add_subparsers(dest="benchmark", required=True)

        t2v = benchmarks.add_parser("t2v", help="evaluate text-to-video outputs")
        t2v.add_argument("--input", default="data/T2V/benchmark_300.jsonl")
        t2v.add_argument("--output", required=True, help="metrics JSONL output")
        t2v.add_argument("--config", default="configs/t2v.yaml")
        t2v.add_argument("--language", choices=("en", "zh"), required=True)
        t2v.add_argument("--max-frames", type=int)
        t2v.add_argument("--debug-dir", "--review-dir", dest="debug_dir")
        t2v.add_argument("--override", action="store_true")
        t2v.add_argument("--project-root", default=".")
        _add_video_arguments(t2v)

        ref2v = benchmarks.add_parser("ref2v", help="evaluate reference-to-video outputs")
        ref2v.add_argument("--input", default="data/Ref2V")
        ref2v.add_argument("--output", required=True, help="evaluator result directory")
        ref2v.add_argument("--config", default="configs/ref2v.yaml")
        ref2v.add_argument("--image-dir", help="reference image directory; inferred from --input by default")
        ref2v.add_argument("--debug-dir")
        ref2v.add_argument("--sample-workers", type=int)
        ref2v.add_argument("--override", action="store_true")
        ref2v.add_argument("--project-root", default=".")
        _add_video_arguments(ref2v)
    return parser


def _run_t2v(args: argparse.Namespace) -> int:
    if args.video and not args.sample_id:
        raise ValueError("--video requires --sample-id")
    config = load_config(args.config)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be a positive integer")
        sampling = config.setdefault("sampling", {})
        sampling["max_frames"] = args.max_frames
        sampling["min_frames"] = min(int(sampling.get("min_frames", 1)), args.max_frames)
    run_t2v(
        args.input,
        args.output,
        config,
        language=args.language,
        video_dir=args.video_dir,
        video_path=args.video,
        video_manifest=args.video_manifest,
        sample_id=args.sample_id,
        override=args.override,
        project_root=Path(args.project_root).resolve(),
        review_root=args.debug_dir,
        debug=bool(args.debug_dir),
    )
    output = Path(args.output)
    root = output.parent.parent if output.parent.name == "passed" else output.parent
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))["evaluator_t2v"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error"] else 0


def _run_ref2v(args: argparse.Namespace) -> int:
    if args.video and not args.sample_id:
        raise ValueError("--video requires --sample-id")
    config = load_config(args.config)
    vlm = Ref2VVLMClient(config) if config.get("llm", {}).get("enabled", True) else None
    summary = run_ref2v(
        args.input,
        args.output,
        config,
        image_dir=args.image_dir,
        video_dir=args.video_dir,
        video_path=args.video,
        video_manifest=args.video_manifest,
        sample_id=args.sample_id,
        project_root=Path(args.project_root).resolve(),
        debug_dir=args.debug_dir,
        vlm=vlm,
        override=args.override,
        sample_workers=args.sample_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["error"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.benchmark == "t2v":
            return _run_t2v(args)
        return _run_ref2v(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"textvbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
