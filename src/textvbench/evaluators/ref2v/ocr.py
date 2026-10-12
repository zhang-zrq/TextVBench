"""Lazy PaddleOCR adapter for Ref2V."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .core import Candidate


class OCRError(RuntimeError):
    """A PaddleOCR dependency or inference failure that must stop evaluation."""


class PaddleOCREngine:
    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        self._ocr = None
        # PaddleOCR/PaddleX is not safe for concurrent first-use or inference.
        # Keep one model instance shared by sample workers, but serialize both
        # initialization and prediction to avoid C++-level races.
        self._predict_lock = threading.Lock()
        self._build_lock = threading.Lock()

    def _build(self):
        if self._ocr is not None:
            return self._ocr
        with self._build_lock:
            if self._ocr is not None:
                return self._ocr
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/tmp/textvbench-paddlex")
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError(
                    "PaddleOCR is required for Ref2V evaluation. Install the OCR runtime "
                    "or pass a test/integration OCR engine programmatically."
                ) from exc
            ocr_cfg = self.config.get("ocr", {})
            kwargs: dict[str, Any] = {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
            if ocr_cfg.get("device"):
                kwargs["device"] = str(ocr_cfg["device"])
            for arg, key in (
                ("text_detection_model_dir", "detection_model_dir"),
                ("text_recognition_model_dir", "recognition_model_dir"),
            ):
                value = ocr_cfg.get(key)
                if value:
                    path = Path(value)
                    if not path.is_absolute():
                        path = self.project_root / path
                    kwargs[arg] = str(path)
            self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def predict(self, image: Any) -> list[Candidate]:
        # PaddleOCR 3.x accepts path strings or numpy arrays, but not pathlib.Path.
        if isinstance(image, Path):
            image = str(image)
        try:
            with self._predict_lock:
                result = next(iter(self._build().predict(image)), None)
            if result is None:
                return []
            data = result if isinstance(result, dict) else getattr(result, "json", lambda: {})()
            if isinstance(data, list):
                data = data[0] if data else {}
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polygons = data.get("rec_polys", data.get("dt_polys", []))
            candidates: list[Candidate] = []
            for index, text in enumerate(texts):
                polygon = polygons[index] if index < len(polygons) else []
                try:
                    bbox = tuple((float(point[0]), float(point[1])) for point in polygon)
                except (TypeError, ValueError, IndexError):
                    bbox = ()
                score = float(scores[index]) if index < len(scores) else 0.0
                candidates.append(Candidate(f"P{index + 1:03d}", str(text), score, bbox))
            return candidates
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"PaddleOCR inference failed: {exc}") from exc
