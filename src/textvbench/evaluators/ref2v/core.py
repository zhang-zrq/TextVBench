"""Pure Ref2V data and metric primitives.

The evaluator deliberately keeps this module independent from PaddleOCR,
OpenAI and OpenCV so metric recomputation and unit tests remain lightweight.
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
    confidence: float = 0.0
    bbox: tuple[tuple[float, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ocr_text": self.text,
            "confidence": float(self.confidence),
            "bbox": [[float(x), float(y)] for x, y in self.bbox],
        }


@dataclass(frozen=True)
class Slot:
    slot_id: str
    target_text: str
    ref_id: str
    carrier_id: str
    reading_order: int
    bbox: tuple[tuple[float, float], ...] = ()


def normalize_text(value: str) -> str:
    return "".join(
        char for char in str(value)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def normalized_edit_distance(left: str, right: str) -> float:
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
    duration = max(float(duration_sec), 0.0)
    if fps <= 0 or min_frames < 1 or max_frames < min_frames:
        raise ValueError("invalid sampling configuration")
    count = max(1, int(math.ceil(duration * fps))) if duration else 1
    count = min(max(count, min_frames), max_frames)
    if count == 1:
        base = [duration / 2.0 if duration else 0.0]
    else:
        step = duration / (count - 1) if duration else 0.0
        base = [index * step for index in range(count)]
    rng = random.Random(stable_seed(seed, sample_id))
    values: list[float] = []
    for index, timestamp in enumerate(base):
        if jitter_sec and duration and index not in (0, count - 1):
            bound = min(abs(float(jitter_sec)), step / 2 if count > 1 else duration / 2)
            timestamp += rng.uniform(-bound, bound)
        values.append(min(max(timestamp, 0.0), duration))
    if duration:
        values[0], values[-1] = 0.0, duration
    return sorted(dict.fromkeys(round(value, 6) for value in values))


def _candidate_value(value: Candidate | dict[str, Any]) -> Candidate:
    if isinstance(value, Candidate):
        return value
    polygon = value.get("bbox") or value.get("polygon") or ()
    return Candidate(
        id=str(value.get("id", value.get("candidate_id", ""))),
        text=str(value.get("ocr_text", value.get("text", ""))),
        confidence=float(value.get("confidence", value.get("score", 0.0))),
        bbox=tuple((float(point[0]), float(point[1])) for point in polygon),
    )


def filter_candidates(
    candidates: Iterable[Candidate | dict[str, Any]],
    *,
    confidence_threshold: float = 0.80,
    max_candidates: int = 20,
    required_texts: Iterable[str] = (),
) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Return active, duplicate-extra and cap-extra candidates."""
    values: list[Candidate] = []
    for item in candidates:
        candidate = _candidate_value(item)
        if candidate.confidence >= confidence_threshold:
            values.append(candidate)
    needed: dict[str, int] = {}
    for text in required_texts:
        key = normalize_text(text)
        if key:
            needed[key] = needed.get(key, 0) + 1
    groups: dict[str, list[Candidate]] = {}
    for candidate in values:
        groups.setdefault(normalize_text(candidate.text), []).append(candidate)
    active: list[Candidate] = []
    duplicate_extra: list[Candidate] = []
    for key, group in groups.items():
        group = sorted(group, key=lambda item: (-item.confidence, item.id))
        keep = max(1, needed.get(key, 0)) if key else len(group)
        active.extend(group[:keep])
        duplicate_extra.extend(group[keep:])
    active.sort(key=lambda item: item.id)
    capped_extra: list[Candidate] = []
    if len(active) > max_candidates:
        ranked = sorted(active, key=lambda item: (-len(normalize_text(item.text)), -item.confidence, item.id))
        active, capped_extra = ranked[:max_candidates], ranked[max_candidates:]
    return active, duplicate_extra, capped_extra


def extra_text_ratio(extra: Iterable[Any], matched: Iterable[Any]) -> float:
    def length(values: Iterable[Any]) -> int:
        total = 0
        for value in values:
            if isinstance(value, Candidate):
                text = value.text
            elif isinstance(value, dict):
                text = value.get("ocr_text", value.get("text", ""))
            else:
                text = value
            total += len(normalize_text(str(text)))
        return total
    extra_len, matched_len = length(extra), length(matched)
    return extra_len / (extra_len + matched_len) if extra_len + matched_len else 0.0


def target_length_weighted_score(scores: dict[str, float], slots: Iterable[Slot | dict[str, Any]]) -> float:
    total = weighted = 0.0
    for slot in slots:
        slot_id = slot.slot_id if isinstance(slot, Slot) else str(slot.get("slot_id", ""))
        text = slot.target_text if isinstance(slot, Slot) else str(slot.get("target_text", ""))
        weight = float(len(normalize_text(text)))
        if weight <= 0:
            continue
        total += weight
        weighted += weight * float(scores.get(slot_id, 0.0))
    return weighted / total if total else 0.0


def harmonic_textv_score(spatial: float, temporal: float) -> float:
    if spatial <= 0 or temporal <= 0:
        return 0.0
    return 100.0 * (2.0 * spatial * temporal / (spatial + temporal))


def bbox_center(bbox: Sequence[Sequence[float]] | tuple[tuple[float, float], ...]) -> tuple[float, float]:
    if not bbox:
        return (float("inf"), float("inf"))
    return (
        sum(float(point[0]) for point in bbox) / len(bbox),
        sum(float(point[1]) for point in bbox) / len(bbox),
    )


def reading_order_score(reference_order: Sequence[str], predicted_order: Sequence[str]) -> float:
    """Pairwise order accuracy with missing slots placed at the end.

    Missing slots retain their reference-relative order. This makes absence a
    measurable ordering error instead of silently removing the slot.
    """
    reference = list(dict.fromkeys(str(item) for item in reference_order))
    if len(reference) < 2:
        return 1.0 if reference else 0.0
    observed = [item for item in predicted_order if item in set(reference)]
    observed = list(dict.fromkeys(observed))
    missing = [item for item in reference if item not in set(observed)]
    effective = observed + missing
    ranks = {item: index for index, item in enumerate(effective)}
    correct = total = 0
    for left_index, left in enumerate(reference):
        for right in reference[left_index + 1:]:
            total += 1
            correct += int(ranks[left] < ranks[right])
    return correct / total if total else 0.0
