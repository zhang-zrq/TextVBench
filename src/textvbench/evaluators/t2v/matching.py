"""Public candidate matching helpers for the T2V evaluator."""

from .core import Candidate, Target, compress_duplicate_candidates, filter_candidates

__all__ = ["Candidate", "Target", "compress_duplicate_candidates", "filter_candidates"]
