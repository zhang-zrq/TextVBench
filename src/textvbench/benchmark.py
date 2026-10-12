"""Benchmark-level metric aggregation shared by the public evaluators."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return sum(values) / len(values) if values else None


def aggregate_benchmark_metrics(
    metrics: Iterable[Mapping[str, Any]],
    *,
    spatial_key: str = "spatial_fidelity",
    temporal_key: str = "temporal_fidelity",
    textv_key: str = "textv_score",
    extra_key: str = "extra_text_ratio",
) -> dict[str, Any]:
    """Return publication-facing means over successful sample metrics.

    ``TextVScore`` is averaged from per-sample scores. It is intentionally not
    recomputed from the means of Spatial and Temporal.
    """
    rows = [row for row in metrics if isinstance(row, Mapping)]
    values = {
        "spatial_fidelity": _mean(rows, spatial_key),
        "temporal_fidelity": _mean(rows, temporal_key),
        "textv_score": _mean(rows, textv_key),
        "extra_text_ratio": _mean(rows, extra_key),
    }
    # Capitalized aliases are convenient for copying into a paper table while
    # snake_case fields remain stable machine-readable names.
    return {
        "Spatial": values["spatial_fidelity"],
        "Temporal": values["temporal_fidelity"],
        "TextVScore": values["textv_score"],
        "Extra": values["extra_text_ratio"],
        **values,
        "sample_count": len(rows),
        "aggregation": "arithmetic_mean_over_successful_samples",
    }
