from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

PROPOSAL_TELEMETRY_METRICS = (
    "local_acceptance_fraction_mean",
    "configuration_esjd_mean",
    "r2_esjd_mean",
    "weighted_free_gap_esjd_mean",
    "invalid_proposal_fraction_max",
    "metropolis_rejection_fraction_max",
)


def summarize_seed_proposal_telemetry(
    seed_results: object,
    *,
    expected_seed_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reduce per-seed proposal telemetry and expose incomplete seed coverage."""

    rows: Sequence[object] = seed_results if isinstance(seed_results, list) else ()
    expected = _expected_seeds(expected_seed_ids)
    values: dict[str, list[float]] = {name: [] for name in PROPOSAL_TELEMETRY_METRICS}
    metric_seeds: dict[str, list[int]] = {name: [] for name in PROPOSAL_TELEMETRY_METRICS}
    recorded_seeds: list[int] = []
    invalid_identity_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            invalid_identity_count += 1
            continue
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            invalid_identity_count += 1
            continue
        recorded_seeds.append(seed)
        dmc_summary = row.get("dmc_summary")
        metadata = dmc_summary.get("metadata") if isinstance(dmc_summary, Mapping) else None
        if not isinstance(metadata, Mapping):
            continue
        for name in PROPOSAL_TELEMETRY_METRICS:
            value = _finite_float(metadata.get(name))
            if value is not None:
                values[name].append(value)
                metric_seeds[name].append(seed)

    if expected is None:
        expected = tuple(dict.fromkeys(recorded_seeds))
    expected_set = set(expected)
    recorded_set = set(recorded_seeds)
    duplicate_seeds = sorted(seed for seed in recorded_set if recorded_seeds.count(seed) > 1)
    unexpected_seeds = sorted(recorded_set - expected_set)
    missing_seeds = sorted(expected_set - recorded_set)
    required_metrics = (
        "local_acceptance_fraction_mean",
        "configuration_esjd_mean",
    )
    missing_metric_seeds = {
        name: sorted(expected_set - set(metric_seeds[name])) for name in required_metrics
    }
    complete = bool(expected) and not (
        invalid_identity_count
        or duplicate_seeds
        or unexpected_seeds
        or missing_seeds
        or any(missing_metric_seeds.values())
    )
    status = (
        "available"
        if complete
        else "telemetry_unavailable"
        if not recorded_seeds
        else "partial_telemetry"
    )

    payload: dict[str, Any] = {
        "status": status,
        "seed_count": len(expected),
        "expected_seed_ids": list(expected),
        "recorded_seed_ids": sorted(recorded_set),
        "missing_seed_ids": missing_seeds,
        "unexpected_seed_ids": unexpected_seeds,
        "duplicate_seed_ids": duplicate_seeds,
        "invalid_seed_identity_count": invalid_identity_count,
        "metric_seed_counts": {name: len(metric_seeds[name]) for name in values},
        "missing_required_metric_seed_ids": missing_metric_seeds,
        "coverage_rule": (
            "acceptance and configuration ESJD require one finite value for every declared seed"
        ),
        "reduction": (
            "unweighted seed mean for mean metrics; maximum across per-seed maxima "
            "for max metrics; observed seed range retained separately"
        ),
        "source": "per-seed DMC proposal metadata",
    }
    for name, samples in values.items():
        if not samples:
            reduced = None
        elif name.endswith("_max"):
            reduced = max(samples)
        else:
            reduced = math.fsum(samples) / len(samples)
        payload[name] = reduced
        payload[f"{name}_seed_min"] = None if not samples else min(samples)
        payload[f"{name}_seed_max"] = None if not samples else max(samples)
    return payload


def _expected_seeds(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    seeds = tuple(value)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("expected_seed_ids must contain unique integer seeds")
    return seeds


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
