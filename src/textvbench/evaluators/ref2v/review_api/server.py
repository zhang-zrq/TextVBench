"""Serve Ref2V debug bundles through a small, read-only HTTP API.

The server intentionally exposes only known evaluator artifacts below the
selected debug/results roots. It is suitable for local review and does not
provide evaluator mutation or arbitrary filesystem access.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlsplit

from ....privacy import redact_debug_value as _redact


COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
FRAME_DIR = re.compile(r"t_[A-Za-z0-9_.-]+\Z")
REFERENCE_IMAGE = re.compile(r"reference_[A-Za-z0-9_.-]+\.jpg\Z")
REFERENCE_OCR = re.compile(r"reference_[A-Za-z0-9_.-]+_ocr\.json\Z")
FRAME_IMAGE = re.compile(r"(?:source_frame|marker|matching_marker_batch_[0-9]+)\.jpg\Z")
FRAME_JSON = re.compile(
    r"(?:observation|ocr_and_filtering|ref_id_matching_vlm_call|matching_vlm_call_batch_[0-9]+)\.json\Z"
)
PROMPT = re.compile(r"[A-Za-z0-9_.-]+\.md\Z")
RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi")
INDEX_FILENAME = ".textvbench_review_index.json"
INDEX_SCHEMA_VERSION = 1


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_object(path: Path) -> dict[str, Any]:
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def safe_path(root: Path, *parts: str) -> Path | None:
    if any(not COMPONENT.fullmatch(part) or part in {".", ".."} for part in parts):
        return None
    candidate = root.joinpath(*parts).resolve()
    return candidate if candidate.is_relative_to(root) else None


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    ocr = config.get("ocr") or {}
    llm = config.get("llm") or {}
    return _redact({
        "sampling": config.get("sampling", {}),
        "ocr": {key: ocr[key] for key in ("confidence_threshold", "max_candidates") if key in ocr},
        "matching": config.get("matching", {}),
        "model": llm.get("active_profile"),
    })


def public_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Normalize Ref2V trace fields and keep every attempt reviewable."""
    attempts = [item for item in (trace.get("attempts") or []) if isinstance(item, dict)]
    final_attempt = attempts[-1] if attempts else {}
    return _redact({
        "call_type": trace.get("call_type"),
        "marker_image": trace.get("marker_image"),
        "payload": trace.get("payload", trace.get("user_payload")),
        "attempts": [
            {
                "attempt": item.get("attempt"),
                "status": item.get("status"),
                "model": item.get("model"),
                "system_prompt": item.get("system_prompt"),
                "response": item.get("response"),
                "error": item.get("error"),
                "recovered_response": item.get("recovered_response"),
                "removed_ids": item.get("removed_ids"),
            }
            for item in attempts
        ],
        "final_prompt": trace.get("final_prompt", final_attempt.get("system_prompt")),
        "final_response": trace.get("response", trace.get("final_response", final_attempt.get("response"))),
        "error": trace.get("error"),
    })


def _frame_folder(timestamp: Any) -> str:
    return f"t_{float(timestamp):.6f}".replace(".", "p")


def _trace_error(path: Path) -> bool:
    value = read_object(path)
    attempts = [item for item in (value.get("attempts") or []) if isinstance(item, dict)]
    return bool(value.get("error")) or bool(attempts and attempts[-1].get("status") not in (None, "ok"))


def _trace_retries(path: Path) -> int:
    attempts = read_object(path).get("attempts") or []
    return sum(
        isinstance(item, dict) and item.get("status") not in (None, "ok")
        for item in attempts
    )


def _bbox_center(boxes: Any) -> tuple[float, float]:
    points = [point for box in (boxes or []) for point in (box or [])]
    if not points:
        return (float("inf"), float("inf"))
    return (
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    )


def _reading_order_evidence(frame: dict[str, Any], references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose the exact missing-at-end pairwise inputs used by the metric."""
    evidence: list[dict[str, Any]] = []
    matches = frame.get("matches") or []
    for reference in references:
        ref_id = str(reference.get("ref_id", ""))
        if ref_id not in (frame.get("matched_ref_ids") or []):
            continue
        slots = sorted(reference.get("slots") or [], key=lambda item: item.get("reading_order", 0))
        expected = [str(item.get("slot_id")) for item in slots]
        positioned: list[tuple[float, float, str]] = []
        for match in matches:
            if str(match.get("ref_id")) != ref_id or not match.get("selected_ids"):
                continue
            center_x, center_y = _bbox_center(match.get("selected_bboxes"))
            positioned.append((center_y, center_x, str(match.get("slot_id"))))
        predicted = list(dict.fromkeys(item[2] for item in sorted(positioned)))
        missing = [slot_id for slot_id in expected if slot_id not in set(predicted)]
        effective = predicted + missing
        ranks = {slot_id: index for index, slot_id in enumerate(effective)}
        comparisons: list[dict[str, Any]] = []
        for left_index, left in enumerate(expected):
            for right in expected[left_index + 1:]:
                comparisons.append({"left": left, "right": right, "correct": ranks[left] < ranks[right]})
        score = (sum(item["correct"] for item in comparisons) / len(comparisons)) if comparisons else (1.0 if expected else 0.0)
        evidence.append({
            "ref_id": ref_id,
            "reference": expected,
            "predicted": predicted,
            "missing": missing,
            "effective": effective,
            "score": score,
            "incorrect_pairs": [item for item in comparisons if not item["correct"]],
        })
    return evidence


class CaseStore:
    """Index Ref2V debug cases and serve normalized review data."""

    def __init__(
        self,
        root: Path,
        *,
        results_root: Path | None = None,
        video_dir: Path | None = None,
        video_manifest: Path | None = None,
    ):
        self.root = root.resolve()
        self.results_root = results_root.resolve() if results_root else None
        self.video_dir = video_dir.resolve() if video_dir else None
        self.video_map = self._load_video_manifest(video_manifest) if video_manifest else {}
        self.index_path = self.root / INDEX_FILENAME
        self._index_lock = threading.RLock()
        self._all_cases: tuple[dict[str, Any], ...] = ()
        self.rebuild_index()

    def rebuild_index(self) -> int:
        """Re-scan evaluator artifacts and atomically replace the persisted index."""
        with self._index_lock:
            cases = tuple(self.summary(sample_id, case) for sample_id, case in self.directories())
            document = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "track": "ref2v",
                "source_root": ".",
                "case_count": len(cases),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "cases": cases,
            }
            if self.root.is_dir():
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=self.root,
                        prefix=f"{INDEX_FILENAME}.",
                        suffix=".tmp",
                        delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        json.dump(document, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.index_path)
                    temporary = None
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            self._all_cases = cases
            return len(cases)

    @staticmethod
    def _load_video_manifest(path: Path | None) -> dict[str, Path]:
        if path is None:
            return {}
        mapping: dict[str, Path] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("sample_id") or not value.get("video_path"):
                raise ValueError(f"video manifest line {line_number} must contain sample_id and video_path")
            sample_id = str(value["sample_id"])
            if sample_id in mapping:
                raise ValueError(f"duplicate video manifest sample_id: {sample_id}")
            video = Path(str(value["video_path"]))
            mapping[sample_id] = video if video.is_absolute() else path.parent / video
        return mapping

    def _case_from_root(self) -> bool:
        return (self.root / "manifest.json").is_file() or (self.root / "result.json").is_file()

    def directories(self) -> Iterable[tuple[str, Path]]:
        if not self.root.is_dir():
            return
        if self._case_from_root():
            yield self.root.name, self.root
            return
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and not path.is_symlink() and COMPONENT.fullmatch(path.name):
                if (path / "manifest.json").is_file() or (path / "result.json").is_file():
                    yield path.name, path

    def case(self, sample_id: str) -> Path | None:
        if self._case_from_root():
            manifest = read_object(self.root / "manifest.json")
            result = read_object(self.root / "result.json")
            root_sample_id = str(manifest.get("sample_id") or result.get("sample_id") or self.root.name)
            return self.root if sample_id == root_sample_id else None
        path = safe_path(self.root, sample_id)
        if path and path.is_dir() and ((path / "manifest.json").is_file() or (path / "result.json").is_file()):
            return path
        return None

    def _public_result(self, sample_id: str) -> dict[str, Any]:
        if self.results_root is None:
            return {}
        return read_object(self.results_root / sample_id / "result.json")

    def _public_error(self, sample_id: str) -> dict[str, Any]:
        if self.results_root is None:
            return {}
        summary = read_object(self.results_root / "summary.json")
        for item in summary.get("errors", []):
            if isinstance(item, dict) and str(item.get("sample_id")) == sample_id:
                return item
        return {}

    def video(self, sample_id: str) -> Path | None:
        path = self.video_map.get(sample_id)
        if path and path.is_file():
            return path
        if self.video_dir:
            for suffix in VIDEO_SUFFIXES:
                candidate = safe_path(self.video_dir, sample_id + suffix)
                if candidate and candidate.is_file():
                    return candidate
        return None

    def _frame_files(self, case: Path, folder: str) -> list[str]:
        path = safe_path(case / "frames", folder)
        if not path or not path.is_dir():
            return []
        allowed: list[str] = []
        for item in path.iterdir():
            if item.is_file() and (FRAME_IMAGE.fullmatch(item.name) or FRAME_JSON.fullmatch(item.name)):
                allowed.append(item.name)
        return sorted(allowed)

    def _references(self, case: Path, manifest: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
        values = manifest.get("references") or result.get("references") or []
        references: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            ref_id = str(item.get("ref_id", ""))
            if not ref_id:
                continue
            ocr = read_object(case / f"reference_{ref_id}_ocr.json")
            image_name = f"reference_{ref_id}.jpg"
            references.append({
                "ref_id": ref_id,
                "filename": item.get("filename"),
                "image": image_name if (case / image_name).is_file() else None,
                "slots": ocr.get("slots", []),
                "carrier_id": item.get("carrier_id"),
                "target_count": item.get("target_count", len(ocr.get("slots", []))),
            })
        if references:
            return references
        for path in sorted(case.glob("reference_*_ocr.json")):
            if not REFERENCE_OCR.fullmatch(path.name):
                continue
            ocr = read_object(path)
            ref_id = str(ocr.get("ref_id", path.stem.removesuffix("_ocr").removeprefix("reference_")))
            references.append({
                "ref_id": ref_id,
                "filename": ocr.get("filename"),
                "image": f"reference_{ref_id}.jpg" if (case / f"reference_{ref_id}.jpg").is_file() else None,
                "slots": ocr.get("slots", []),
                "carrier_id": None,
                "target_count": len(ocr.get("slots", [])),
            })
        return references

    def _frames(self, case: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
        values = result.get("frames") or []
        frames: list[dict[str, Any]] = []
        for position, item in enumerate(values):
            if not isinstance(item, dict) or not isinstance(item.get("timestamp"), (int, float)):
                continue
            folder = _frame_folder(item["timestamp"])
            # The evaluator currently uses timestamp-only directories. Search
            # by prefix for future bundles that add a frame index suffix.
            if not (case / "frames" / folder).is_dir():
                matches = sorted(case.glob(f"frames/{folder}_*"))
                folder = matches[0].name if matches else folder
            observation = read_object(case / "frames" / folder / "observation.json")
            merged = {**observation, **item}
            filtering = read_object(case / "frames" / folder / "ocr_and_filtering.json")
            merged["filtering"] = {
                "active_ids": [str(value.get("id")) for value in filtering.get("active_candidates", []) if isinstance(value, dict)],
                "duplicate_extra_ids": [str(value.get("id")) for value in filtering.get("duplicate_extras", []) if isinstance(value, dict)],
                "capped_extra_ids": [str(value.get("id")) for value in filtering.get("capped_extras", []) if isinstance(value, dict)],
            }
            merged["folder"] = folder
            merged["position"] = position
            merged["files"] = self._frame_files(case, folder)
            frames.append(merged)
        if not frames:
            for path in sorted((case / "frames").glob("t_*")) if (case / "frames").is_dir() else []:
                if not path.is_dir() or not FRAME_DIR.fullmatch(path.name):
                    continue
                observation = read_object(path / "observation.json")
                if observation:
                    filtering = read_object(path / "ocr_and_filtering.json")
                    observation["filtering"] = {
                        "active_ids": [str(value.get("id")) for value in filtering.get("active_candidates", []) if isinstance(value, dict)],
                        "duplicate_extra_ids": [str(value.get("id")) for value in filtering.get("duplicate_extras", []) if isinstance(value, dict)],
                        "capped_extra_ids": [str(value.get("id")) for value in filtering.get("capped_extras", []) if isinstance(value, dict)],
                    }
                    observation["folder"] = path.name
                    observation["position"] = len(frames)
                    observation["files"] = self._frame_files(case, path.name)
                    frames.append(observation)
        return frames

    def summary(self, sample_id: str, case: Path) -> dict[str, Any]:
        manifest = read_object(case / "manifest.json")
        debug_result = read_object(case / "result.json")
        public_result = self._public_result(sample_id)
        result = public_result or debug_result
        metrics = public_result.get("metrics") or debug_result.get("metrics") or manifest.get("metrics") or {}
        frames = self._frames(case, debug_result)
        trace_files = [
            path for path in case.glob("frames/*/*.json")
            if FRAME_JSON.fullmatch(path.name) and "vlm_call" in path.name
        ]
        error = self._public_error(sample_id)
        status = "complete" if result and metrics else "error"
        video_info = manifest.get("video") or result.get("video") or public_result.get("video") or {}
        return {
            "sample_id": str(manifest.get("sample_id") or result.get("sample_id") or sample_id),
            "status": status,
            "metrics": _redact(metrics),
            "duration_sec": video_info.get("duration_sec"),
            "video_available": self.video(sample_id) is not None,
            "frame_count": len(frames),
            "reference_count": len(self._references(case, manifest, result)),
            "evaluated_frames": sum(bool(frame.get("evaluated")) for frame in frames),
            "no_match_frames": sum(not bool(frame.get("evaluated")) for frame in frames),
            "trace_errors": sum(_trace_error(path) for path in trace_files),
            "vlm_retries": sum(_trace_retries(path) for path in trace_files),
            "error": _redact(error) if error else None,
        }

    def list_cases(self, query: dict[str, list[str]]) -> dict[str, Any]:
        with self._index_lock:
            all_cases = self._all_cases
        cases = list(all_cases)
        search = query.get("search", [""])[0].strip().lower()
        status = query.get("status", [""])[0]
        cases = [item for item in cases if
                 (not search or search in item["sample_id"].lower()) and
                 (not status or item["status"] == status)]
        sort = query.get("sort", ["score_asc"])[0]
        if sort not in {"score_asc", "score_desc", "spatial_asc", "temporal_asc", "extra_desc", "reading_asc", "id"}:
            raise ValueError("invalid sort")
        if sort == "id":
            cases.sort(key=lambda item: item["sample_id"])
        else:
            metric = {
                "spatial_asc": "length_weighted_spatial_fidelity",
                "temporal_asc": "length_weighted_temporal_fidelity",
                "extra_desc": "extra_text_ratio",
                "reading_asc": "reading_order_score",
            }.get(sort, "textv_score")
            descending = sort == "score_desc" or sort == "extra_desc"
            cases.sort(key=lambda item: (
                item["metrics"].get(metric) is None,
                -(item["metrics"].get(metric) or 0.0) if descending else (item["metrics"].get(metric) or 0.0),
                item["sample_id"],
            ))
        try:
            page = int(query.get("page", ["1"])[0])
            page_size = int(query.get("page_size", ["20"])[0])
        except ValueError as exc:
            raise ValueError("page and page_size must be integers") from exc
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("page must be positive and page_size must be 1..100")
        start = (page - 1) * page_size
        scored = [item["metrics"].get("textv_score") for item in all_cases if item["metrics"].get("textv_score") is not None]
        return {
            "items": cases[start:start + page_size], "total": len(cases), "page": page, "page_size": page_size,
            "counts": {
                "cases": len(all_cases), "error": sum(item["status"] == "error" for item in all_cases),
                "average_score": sum(scored) / len(scored) if scored else None,
            },
        }

    def detail(self, sample_id: str, case: Path) -> dict[str, Any]:
        manifest = read_object(case / "manifest.json")
        debug_result = read_object(case / "result.json")
        public_result = self._public_result(sample_id)
        result = public_result or debug_result
        frames = self._frames(case, debug_result)
        references = self._references(case, manifest, result)
        for frame in frames:
            frame["reading_order"] = _reading_order_evidence(frame, references)
        public_manifest = _redact(manifest)
        public_manifest.pop("video_path", None)
        public_result_value = _redact({key: value for key, value in result.items() if key not in {"video_path", "frames"}})
        return {
            "summary": self.summary(sample_id, case),
            "manifest": public_manifest,
            "input_record": _redact(read_object(case / "input_record.json")),
            "config": public_config(read_object(case / "resolved_config.json")),
            "result": public_result_value,
            "references": references,
            "frames": [_redact(frame) for frame in frames],
            "run_details": _redact(read_object(self.results_root / "run_details.json")) if self.results_root else {},
        }


class Handler(BaseHTTPRequestHandler):
    store: CaseStore
    static_root: Path

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_video(self, path: Path) -> None:
        size = path.stat().st_size
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header:
            match = RANGE.fullmatch(header)
            if not match or (not match.group(1) and not match.group(2)):
                return self.send_json({"error": "invalid range"}, 416)
            if match.group(1):
                start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), end)
            else:
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                return self.send_json({"error": "range not satisfiable"}, 416)
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        parts = path.strip("/").split("/") if path.strip("/") else []
        query = parse_qs(urlsplit(self.path).query)
        try:
            if path.rstrip("/") == "/api/health":
                return self.send_json({"status": "ok", "cases": self.store.list_cases({})["total"]})
            if path.rstrip("/") == "/api/cases":
                if query.get("refresh", [""])[0] == "1":
                    self.store.rebuild_index()
                return self.send_json(self.store.list_cases(query))
            if len(parts) >= 3 and parts[:2] == ["api", "cases"]:
                sample_id = parts[2]
                case = self.store.case(sample_id)
                if case is None:
                    return self.send_json({"error": "case not found"}, 404)
                if len(parts) == 3:
                    return self.send_json(self.store.detail(sample_id, case))
                if len(parts) == 6 and parts[3] == "frames" and FRAME_DIR.fullmatch(parts[4]) and FRAME_JSON.fullmatch(parts[5]):
                    artifact = safe_path(case / "frames", parts[4], parts[5])
                    if artifact and artifact.is_file():
                        data = read_object(artifact)
                        return self.send_json(public_trace(data) if "vlm_call" in parts[5] else _redact(data))
                if len(parts) == 5 and parts[3] == "references" and REFERENCE_OCR.fullmatch(parts[4]):
                    artifact = safe_path(case, parts[4])
                    if artifact and artifact.is_file():
                        return self.send_json(_redact(read_object(artifact)))
                return self.send_json({"error": "artifact not found"}, 404)
            if len(parts) >= 3 and parts[:2] == ["media", "cases"]:
                sample_id = parts[2]
                case = self.store.case(sample_id)
                if case is None:
                    return self.send_json({"error": "case not found"}, 404)
                if len(parts) == 4 and parts[3] == "video":
                    video = self.store.video(sample_id)
                    if video:
                        return self.send_video(video)
                if len(parts) == 5 and parts[3] == "references" and REFERENCE_IMAGE.fullmatch(parts[4]):
                    artifact = safe_path(case, parts[4])
                    if artifact and artifact.is_file():
                        return self.send_file(artifact)
                if len(parts) == 6 and parts[3] == "frames" and FRAME_DIR.fullmatch(parts[4]) and FRAME_IMAGE.fullmatch(parts[5]):
                    artifact = safe_path(case / "frames", parts[4], parts[5])
                    if artifact and artifact.is_file():
                        return self.send_file(artifact)
                return self.send_json({"error": "artifact not found"}, 404)
            if path.startswith("/api/") or path.startswith("/media/"):
                return self.send_json({"error": "not found"}, 404)
            relative = path.lstrip("/") or "index.html"
            target = (self.static_root / relative).resolve()
            if target.is_file() and target.is_relative_to(self.static_root):
                return self.send_file(target)
            return self.send_json({"error": "not found"}, 404)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self.send_json({"error": _redact(str(exc), key="error")}, 400)


class ReviewHTTPServer(ThreadingHTTPServer):
    """Use a browser-sized accept queue for parallel images, JSON, and video."""

    daemon_threads = True
    request_queue_size = 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only TextVBench Ref2V evaluator review")
    parser.add_argument("--data", required=True, help="Ref2V debug root or one sample debug directory")
    parser.add_argument("--results-dir", help="public evaluator output containing summary/run_details and sample results")
    parser.add_argument("--video-dir", help="directory containing videos named <sample_id>.<suffix>")
    parser.add_argument("--video-manifest", help="JSONL mapping sample_id to video_path")
    parser.add_argument("--frontend", default="frontend/ref2v_review")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args(argv)
    Handler.store = CaseStore(
        Path(args.data),
        results_root=Path(args.results_dir) if args.results_dir else None,
        video_dir=Path(args.video_dir) if args.video_dir else None,
        video_manifest=Path(args.video_manifest) if args.video_manifest else None,
    )
    Handler.static_root = Path(args.frontend).resolve()
    server = ReviewHTTPServer((args.host, args.port), Handler)
    print(f"Indexed {len(Handler.store._all_cases)} cases at {Handler.store.index_path}", flush=True)
    print(f"Ref2V review: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
