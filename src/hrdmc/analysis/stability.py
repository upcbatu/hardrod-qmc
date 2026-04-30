from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def summarize_replicate_metrics(
    rows: Sequence[Mapping[str, float]],
    metric_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    if not rows:
        raise ValueError("at least one replicate row is required")
    if not metric_names:
        raise ValueError("at least one metric name is required")

    summaries: dict[str, dict[str, float | int]] = {}
    for metric_name in metric_names:
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"metric contains non-finite values: {metric_name}")
        sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summaries[metric_name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "sample_std": sample_std,
            "stderr": sample_std / float(np.sqrt(values.size)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "spread": float(np.max(values) - np.min(values)),
        }
    return summaries
