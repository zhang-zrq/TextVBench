"""Serve evaluator review bundles without exposing arbitrary local files."""

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
from urllib.parse import parse_qs, unquote, urlsplit

from ....privacy import redact_debug_value


COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
IMAGE = re.compile(r"(?:source_frame|carrier_marker|matching_marker_batch_[0-9]+)\.jpg\Z")
TRACE = re.compile(r"(?:carrier_vlm_call|matching_vlm_call_batch_[0-9]+|ocr_and_filtering|observation)\.json\Z")
RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")
INDEX_FILENAME = ".textvbench_review_index.json"
INDEX_SCHEMA_VERSION = 1


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def safe_path(root: Path, *parts: str) -> Path | None:
    if any(not COMPONENT.fullmatch(part) or part in {".", ".."} for part in parts):
        return None
    candidate = root.joinpath(*parts).resolve()
    return candidate if candidate.is_relative_to(root) else None


def public_config(config: dict) -> dict:
    ocr = config.get("ocr") or {}
    llm = config.get("llm") or {}
    return {
        "sampling": config.get("sampling", {}),
        "ocr": {key: ocr[key] for key in ("confidence_threshold", "max_candidates") if key in ocr},
        "matching": config.get("matching", {}),
        "model": llm.get("active_profile"),
    }


def public_trace(trace: dict) -> dict:
    # Preserve the actual prompt, payload and response; omit backend details.
    return redact_debug_value({
        "call_type": trace.get("call_type"),
        "marker_image": trace.get("marker_image"),
        "user_payload": trace.get("user_payload"),
        "attempts": [
            {
                "attempt": attempt.get("attempt"), "status": attempt.get("status"),
                "model": attempt.get("model"), "system_prompt": attempt.get("system_prompt"),
                "response": attempt.get("response"), "error": attempt.get("error"),
                "recovered_response": attempt.get("recovered_response"),
                "removed_ids": attempt.get("removed_ids"),
            }
            for attempt in (trace.get("attempts") or []) if isinstance(attempt, dict)
        ],
        "final_prompt": trace.get("final_prompt"),
        "final_response": trace.get("final_response"),
        "error": trace.get("error"),
    })


class CaseStore:
    def __init__(self, root: Path, video_dir: Path | None = None):
        self.root = root.resolve()
        self.video_dir = video_dir.resolve() if video_dir else None
        self.index_path = self.root / INDEX_FILENAME
        self._index_lock = threading.RLock()
        self._all_cases: tuple[dict, ...] = ()
        self.rebuild_index()

    def rebuild_index(self) -> int:
        """Re-scan evaluator artifacts and atomically replace the persisted index."""
        with self._index_lock:
            cases = tuple(
                self.summary(benchmark, case_id, case)
                for benchmark, case_id, case in self.directories()
            )
            document = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "track": "t2v",
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

    def directories(self):
        if not self.root.is_dir():
            return
        # Both tests/evaluator_review and tests/evaluator_review/t2v are accepted.
        if any(child.is_dir() and ((child / "manifest.json").is_file() or (child / "result.json").is_file())
               for child in self.root.iterdir()):
            benchmarks = [self.root]
        else:
            benchmarks = [child for child in self.root.iterdir() if child.is_dir()]
        for benchmark in sorted(benchmarks):
            if not COMPONENT.fullmatch(benchmark.name) or benchmark.is_symlink():
                continue
            for case in sorted(benchmark.iterdir()):
                if case.is_dir() and not case.is_symlink() and COMPONENT.fullmatch(case.name):
                    if (case / "manifest.json").is_file() or (case / "result.json").is_file():
                        yield benchmark.name, case.name, case

    def case(self, benchmark: str, case_id: str) -> Path | None:
        if self.root.name == benchmark and self.root.is_dir():
            path = safe_path(self.root, case_id)
        else:
            path = safe_path(self.root, benchmark, case_id)
        if path and path.is_dir() and ((path / "manifest.json").is_file() or (path / "result.json").is_file()):
            return path
        return None

    def video(self, manifest: dict) -> Path | None:
        sample_id = manifest.get("sample_id", "")
        if not self.video_dir or not isinstance(sample_id, str):
            return None
        path = safe_path(self.video_dir, sample_id + ".mp4")
        return path if path and path.is_file() else None

    def summary(self, benchmark: str, case_id: str, case: Path) -> dict:
        manifest = read_object(case / "manifest.json")
        result = read_object(case / "result.json")
        metrics = result.get("metrics") or manifest.get("metrics") or {}
        complete = bool(result and result.get("metrics"))
        return {
            "benchmark": benchmark, "case_id": case_id,
            "sample_id": manifest.get("sample_id") or result.get("sample_id") or case_id,
            "pair_id": read_object(case / "input_record.json").get("pair_id"),
            "source_qc": read_object(case / "input_record.json").get("qc"),
            "language": manifest.get("language") or result.get("language"),
            "status": "complete" if complete else "error",
            "metrics": metrics,
            "duration_sec": (manifest.get("video") or result.get("video") or {}).get("duration_sec"),
            "frame_count": len(result.get("frames") or []),
            "video_available": self.video(manifest) is not None,
        }

    def list_cases(self, query: dict) -> dict:
        with self._index_lock:
            all_cases = self._all_cases
        q = query.get("search", [""])[0].strip().lower()
        items = [item for item in all_cases if
                 (not q or q in str(item["sample_id"]).lower() or q in str(item["pair_id"] or "").lower())
                 and all(not query.get(field, [""])[0] or str(item[field]) == query[field][0]
                         for field in ("benchmark", "language", "status"))]
        sort = query.get("sort", ["score_asc"])[0]
        if sort not in {"score_asc", "score_desc", "extra_desc", "id"}:
            raise ValueError("invalid sort")
        if sort == "id":
            items.sort(key=lambda item: (item["sample_id"], item["language"] or ""))
        else:
            metric = "extra_text_ratio" if sort == "extra_desc" else "textv_score"
            descending = sort != "score_asc"
            items.sort(key=lambda item: (item["metrics"].get(metric) is None,
                       -(item["metrics"].get(metric) or 0) if descending else (item["metrics"].get(metric) or 0),
                       item["case_id"]))
        try:
            page = int(query.get("page", ["1"])[0])
            page_size = int(query.get("page_size", ["20"])[0])
        except ValueError as exc:
            raise ValueError("page and page_size must be integers") from exc
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("page must be positive and page_size must be 1..100")
        start = (page - 1) * page_size
        return {"items": items[start:start + page_size], "total": len(items),
                "page": page, "page_size": page_size,
                "facets": {"benchmarks": sorted({c["benchmark"] for c in all_cases}),
                           "languages": sorted({c["language"] for c in all_cases if c["language"]})},
                "counts": {"cases": len(all_cases), "error": sum(c["status"] == "error" for c in all_cases),
                           "low_score": sum((c["metrics"].get("textv_score") or 0) < 60
                                            for c in all_cases if c["metrics"].get("textv_score") is not None),
                           "average_score": (sum(c["metrics"]["textv_score"] for c in all_cases
                                                 if c["metrics"].get("textv_score") is not None) /
                                             sum(c["metrics"].get("textv_score") is not None for c in all_cases))
                           if any(c["metrics"].get("textv_score") is not None for c in all_cases) else None}}

    def detail(self, benchmark: str, case_id: str, case: Path) -> dict:
        manifest = read_object(case / "manifest.json")
        result = read_object(case / "result.json")
        frames = []
        for frame in result.get("frames") or []:
            timestamp = frame.get("timestamp")
            index = frame.get("frame_index")
            if not isinstance(timestamp, (int, float)) or not isinstance(index, int):
                continue
            folder = f"t_{timestamp:.6f}".replace(".", "p") + f"_frame_{index:05d}"
            frame_path = safe_path(case / "frames", folder)
            files = sorted(path.name for path in frame_path.iterdir() if path.is_file()) if frame_path and frame_path.is_dir() else []
            frames.append({**frame, "folder": folder, "files": files})
        # Absolute local video paths are never sent to the browser.
        manifest = {key: value for key, value in manifest.items() if key != "video_path"}
        result = {key: value for key, value in result.items() if key not in {"video_path", "frames"}}
        return {"summary": self.summary(benchmark, case_id, case), "manifest": manifest,
                "result": result, "frames": frames,
                "input_record": read_object(case / "input_record.json"),
                "config": public_config(read_object(case / "resolved_config.json"))}


class Handler(BaseHTTPRequestHandler):
    store: CaseStore
    static_root: Path

    def send_json(self, value: dict, status: int = 200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_video(self, path: Path):
        size = path.stat().st_size
        header = self.headers.get("Range")
        start, end = 0, size - 1
        if header:
            match = RANGE.fullmatch(header)
            if not match or not match.group(1) and not match.group(2):
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
        self.send_header("Content-Type", "video/mp4")
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

    @staticmethod
    def _api_path(path: str) -> str:
        """Strip an optional reverse-proxy prefix before routing API/media."""
        for marker in ("/api/", "/media/"):
            index = path.find(marker)
            if index >= 0:
                return path[index:]
        return path

    def do_GET(self):  # noqa: N802
        path = self._api_path(unquote(urlsplit(self.path).path))
        parts = path.strip("/").split("/")
        query = parse_qs(urlsplit(self.path).query)
        if path.rstrip("/") == "/api/cases":
            try:
                if query.get("refresh", [""])[0] == "1":
                    self.store.rebuild_index()
                return self.send_json(self.store.list_cases(query))
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, 400)
        if path.rstrip("/") == "/api/health":
            return self.send_json({"status": "ok", "cases": self.store.list_cases({})["total"]})
        if len(parts) >= 4 and parts[:2] in (["api", "cases"], ["media", "cases"]):
            benchmark, case_id = parts[2:4]
            case = self.store.case(benchmark, case_id)
            if not case:
                return self.send_json({"error": "case not found"}, 404)
            if parts[0] == "api":
                if len(parts) == 4:
                    return self.send_json(self.store.detail(benchmark, case_id, case))
                if len(parts) == 7 and parts[4] == "frames" and TRACE.fullmatch(parts[6]):
                    frame_path = safe_path(case / "frames", parts[5], parts[6])
                    if frame_path and frame_path.is_file():
                        data = read_object(frame_path)
                        return self.send_json(public_trace(data) if "vlm_call" in parts[6] else data)
            elif len(parts) == 7 and parts[4] == "frames" and IMAGE.fullmatch(parts[6]):
                image_path = safe_path(case / "frames", parts[5], parts[6])
                if image_path and image_path.is_file():
                    return self.send_file(image_path)
            elif len(parts) == 5 and parts[4] == "video":
                video = self.store.video(read_object(case / "manifest.json"))
                if video:
                    return self.send_video(video)
            return self.send_json({"error": "artifact not found"}, 404)
        if path.startswith("/api/") or path.startswith("/media/"):
            return self.send_json({"error": "not found"}, 404)
        relative = path.lstrip("/") or "index.html"
        target = (self.static_root / relative).resolve()
        if target.is_file() and target.is_relative_to(self.static_root):
            return self.send_file(target)
        return self.send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Read-only TextVBench evaluator review")
    parser.add_argument("--data", required=True, help="Bundle root or benchmark folder (pass explicitly)")
    parser.add_argument("--frontend", default="frontend/t2v_review")
    parser.add_argument("--video-dir", help="Opt-in directory with <sample_id>.mp4 files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    Handler.store = CaseStore(Path(args.data), Path(args.video_dir) if args.video_dir else None)
    Handler.static_root = Path(args.frontend).resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Indexed {len(Handler.store._all_cases)} cases at {Handler.store.index_path}", flush=True)
    print(f"Evaluator review: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
