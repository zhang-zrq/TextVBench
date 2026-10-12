"""Pure T2V evaluator algorithms.

This module intentionally has no OpenCV, PaddleOCR or OpenAI dependency.  It
is the stable, unit-testable contract shared by the video evaluator and future
offline metric tooling.
"""

from __future__ import annotations

import hashlib
import math
import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ocr_text": self.text,
            "confidence": float(self.confidence),
            "bbox": [[float(x), float(y)] for x, y in self.bbox],
        }


@dataclass(frozen=True)
class Target:
    slot_id: str
    text: str
    carrier_id: str
    carrier: str = ""
    position: str = ""


def normalize_text(value: str) -> str:
    """Remove whitespace and punctuation while retaining case and CJK chars."""
    return "".join(
        ch for ch in str(value)
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def normalized_edit_distance(left: str, right: str) -> float:
    """Character-level normalized Levenshtein distance in ``[0, 1]``."""
    a, b = normalize_text(left), normalize_text(right)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1] / max(len(a), len(b))


def normalized_edit_similarity(left: str, right: str) -> float:
    return 1.0 - normalized_edit_distance(left, right)


def stable_seed(global_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return (int(global_seed) + int.from_bytes(digest[:8], "big")) % (2**32)


def sample_timestamps(
    duration_sec: float,
    *,
    fps: float = 3.0,
    min_frames: int = 1,
    max_frames: int = 30,
    seed: int = 42,
    sample_id: str = "",
    jitter_sec: float = 0.0,
) -> list[float]:
    """Create deterministic timestamp samples with endpoint coverage.

    Jitter is bounded so timestamps stay inside the video and are sorted.  A
    non-positive duration still returns ``[0.0]`` when ``min_frames`` allows
    one frame, which makes corrupt/empty metadata diagnosable downstream.
    """
    duration = max(float(duration_sec), 0.0)
    if fps <= 0 or min_frames < 1 or max_frames < min_frames:
        raise ValueError("invalid sampling configuration")
    count = max(1, int(math.ceil(duration * fps))) if duration else 1
    count = min(max(count, min_frames), max_frames)
    if count == 1:
        base = [0.0 if not duration else duration / 2.0]
    else:
        step = duration / (count - 1) if duration else 0.0
        base = [i * step for i in range(count)]
    rng = random.Random(stable_seed(seed, sample_id))
    timestamps = []
    for index, timestamp in enumerate(base):
        if jitter_sec and duration:
            # Keep the first/last samples close to their required endpoints.
            bound = min(abs(float(jitter_sec)), (base[1] - base[0]) / 2 if count > 1 else duration / 2)
            if index not in (0, count - 1):
                timestamp += rng.uniform(-bound, bound)
        timestamps.append(min(max(timestamp, 0.0), duration))
    timestamps[0] = 0.0 if duration else 0.0
    if duration and len(timestamps) > 1:
        timestamps[-1] = duration
    return sorted(dict.fromkeys(round(t, 6) for t in timestamps))


def _candidate_value(candidate: Candidate | dict[str, Any]) -> Candidate:
    if isinstance(candidate, Candidate):
        return candidate
    bbox = candidate.get("bbox") or candidate.get("polygon") or ()
    return Candidate(
        id=str(candidate.get("id", candidate.get("candidate_id", ""))),
        text=str(candidate.get("ocr_text", candidate.get("text", ""))),
        confidence=float(candidate.get("confidence", 0.0)),
        bbox=tuple((float(point[0]), float(point[1])) for point in bbox),
    )


def filter_candidates(
    candidates: Iterable[Candidate | dict[str, Any]],
    *,
    confidence_threshold: float = 0.80,
    max_candidates: int = 20,
    required_texts: Iterable[str] = (),
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Return ``(active, duplicate_extra, capped_extra)`` candidates.

    Confidence-filtered candidates are excluded entirely.  Duplicate text is
    compressed unless the GT needs multiple instances of that normalized text.
    """
    values = []
    for item in candidates:
        candidate = _candidate_value(item)
        if candidate.confidence >= confidence_threshold:
            values.append(candidate)
    needed: dict[str, int] = {}
    for text in required_texts:
        key = normalize_text(text)
        if key:
            needed[key] = needed.get(key, 0) + 1
    by_text: dict[str, list[Candidate]] = {}
    for candidate in values:
        by_text.setdefault(normalize_text(candidate.text), []).append(candidate)
    active: list[Candidate] = []
    duplicate_extra: list[Candidate] = []
    for key, group in by_text.items():
        group = sorted(group, key=lambda item: (-item.confidence, item.id))
        keep = max(1, needed.get(key, 0)) if key else len(group)
        active.extend(group[:keep])
        duplicate_extra.extend(group[keep:])
    active.sort(key=lambda item: item.id)
    capped_extra: list[Candidate] = []
    if len(active) > max_candidates:
        ranked = sorted(active, key=lambda item: (-len(normalize_text(item.text)), -item.confidence, item.id))
        active = ranked[:max_candidates]
        capped_extra = ranked[max_candidates:]
    return active, duplicate_extra, capped_extra


def compress_duplicate_candidates(
    candidates: Iterable[Candidate | dict[str, Any]],
    *,
    required_texts: Iterable[str] = (),
) -> tuple[list[Candidate], list[Candidate]]:
    """Compatibility helper returning active and duplicate extras only."""
    active, duplicate_extra, _ = filter_candidates(
        candidates, confidence_threshold=0.0, max_candidates=10**9, required_texts=required_texts
    )
    return active, duplicate_extra


def extra_text_ratio(extra: Iterable[str | Candidate | dict[str, Any]], matched: Iterable[str | Candidate | dict[str, Any]]) -> float:
    def length(items: Iterable[str | Candidate | dict[str, Any]]) -> int:
        total = 0
        for item in items:
            value = item.text if isinstance(item, Candidate) else item.get("ocr_text", item.get("text", "")) if isinstance(item, dict) else item
            total += len(normalize_text(str(value)))
        return total
    extra_len, matched_len = length(extra), length(matched)
    return extra_len / (extra_len + matched_len) if extra_len + matched_len else 0.0


def target_length_weighted_score(scores: dict[str, float], targets: Iterable[Target | dict[str, Any]]) -> float:
    total, weighted = 0.0, 0.0
    for target in targets:
        slot_id = target.slot_id if isinstance(target, Target) else str(target.get("slot_id", ""))
        text = target.text if isinstance(target, Target) else str(target.get("target_text", target.get("text", target.get("text_en", target.get("text_zh", "")))))
        weight = float(len(normalize_text(text)))
        if weight <= 0:
            continue
        total += weight
        weighted += weight * float(scores.get(slot_id, 0.0))
    return weighted / total if total else 0.0


def harmonic_textv_score(spatial: float, temporal: float) -> float:
    if spatial <= 0 or temporal <= 0:
        return 0.0
    return 100.0 * (2 * spatial * temporal / (spatial + temporal))
