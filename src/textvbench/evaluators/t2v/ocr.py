"""PaddleOCR adapter used by the T2V evaluator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .core import Candidate


class OCRError(RuntimeError):
    """A PaddleOCR dependency or inference failure that must stop evaluation."""


class PaddleOCREngine:
    """Lazy PaddleOCR wrapper with a small, version-tolerant result adapter."""

    def __init__(self, config: dict[str, Any], project_root: Path | None = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        self._ocr = None

    def _build(self):
        if self._ocr is not None:
            return self._ocr
        # PaddleX reads this environment variable during import.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path("/tmp/paddlex")))
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "PaddleOCR is required for T2V evaluation. Install the optional "
                "ocr extra and a PaddlePaddle wheel matching the CUDA runtime."
            ) from exc
        kwargs: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        ocr_config = self.config.get("ocr", {})
        if ocr_config.get("device"):
            kwargs["device"] = str(ocr_config["device"])
        for key, option in (
            ("text_detection_model_dir", "detection_model_dir"),
            ("text_recognition_model_dir", "recognition_model_dir"),
        ):
            value = ocr_config.get(option)
            if value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = self.project_root / candidate
                kwargs[key] = str(candidate)
        self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def predict(self, frame: Any) -> list[Candidate]:
        try:
            result = next(iter(self._build().predict(frame)), None)
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
