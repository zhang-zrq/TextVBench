"""TextVBench-T2V visual-text evaluator."""

from .core import (
    Candidate,
    Target,
    compress_duplicate_candidates,
    extra_text_ratio,
    normalized_edit_similarity,
    normalize_text,
    sample_timestamps,
    target_length_weighted_score,
)
from .pipeline import evaluate_record, run

__all__ = [
    "Candidate",
    "Target",
    "compress_duplicate_candidates",
    "evaluate_record",
    "extra_text_ratio",
    "normalized_edit_similarity",
    "normalize_text",
    "run",
    "sample_timestamps",
    "target_length_weighted_score",
]
