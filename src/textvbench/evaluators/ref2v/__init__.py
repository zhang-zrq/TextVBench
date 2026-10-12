"""Independent TextVBench-Ref2V evaluator."""

from .core import (
    Candidate,
    Slot,
    extra_text_ratio,
    harmonic_textv_score,
    normalized_edit_similarity,
    normalize_text,
    reading_order_score,
    target_length_weighted_score,
)
from .pipeline import evaluate_record, run

__all__ = [
    "Candidate",
    "Slot",
    "extra_text_ratio",
    "harmonic_textv_score",
    "normalized_edit_similarity",
    "normalize_text",
    "reading_order_score",
    "target_length_weighted_score",
    "evaluate_record",
    "run",
]
