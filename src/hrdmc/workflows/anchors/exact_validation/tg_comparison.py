from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from hrdmc.monte_carlo.dmc.local import DMCStreamingSummary
from hrdmc.theory import (
    trapped_tg_density_profile,
    trapped_tg_density_profile_semiclassical,
    trapped_tg_r2_radius,
    trapped_tg_rms_radius,
)
from hrdmc.workflows.anchors.exact_validation.models import TrappedTGAnchor


def density_profile_payload(
    anchor: TrappedTGAnchor,
    seed_summaries: list[DMCStreamingSummary],
    pure_summary: dict[str, Any],
) -> dict[str, Any]:
    first = seed_summaries[0]
    edges = first.density_bin_edges
    widths = np.diff(edges)
    x = 0.5 * (edges[:-1] + edges[1:])
    mixed_by_seed = np.asarray([summary.density for summary in seed_summaries])
    mixed = np.mean(mixed_by_seed, axis=0)
    exact = trapped_tg_density_profile(x, n_particles=anchor.n_particles, omega=anchor.omega)
    exact_bin_average = _bin_averaged_trapped_tg_density(
        edges,
        n_particles=anchor.n_particles,
        omega=anchor.omega,
    )
    large_n = trapped_tg_density_profile_semiclassical(
        x,
        n_particles=anchor.n_particles,
        omega=anchor.omega,
    )
    payload: dict[str, Any] = {
        "x": x.tolist(),
        "bin_edges": edges.tolist(),
        "mixed_n_x": mixed.tolist(),
        "mixed_seed_stderr": _finite_list_or_none(_density_stderr(mixed_by_seed)),
        "mixed_integral": float(np.sum(mixed * widths)),
        "exact_n_x": exact.tolist(),
        "exact_integral": float(np.sum(exact * widths)),
        "exact_bin_averaged_n_x": exact_bin_average.tolist(),
        "exact_bin_averaged_integral": float(np.sum(exact_bin_average * widths)),
        "exact_tg_large_n_n_x": large_n.tolist(),
        "exact_tg_large_n_integral": float(np.sum(large_n * widths)),
    }
    pure_density = pure_summary.get("observables", {}).get("density")
    if isinstance(pure_density, dict):
        pure_values = np.asarray(pure_density.get("value", []), dtype=float)
        if pure_values.shape == exact.shape and np.all(np.isfinite(pure_values)):
            payload["pure_fw_n_x"] = pure_values.tolist()
            payload["pure_fw_seed_stderr"] = pure_density.get("stderr")
            payload["pure_fw_integral"] = float(np.sum(pure_values * widths))
    return payload


def trapped_tg_exact_comparison(
    anchor: TrappedTGAnchor,
    *,
    seed_summaries: list[DMCStreamingSummary],
    pure_summary: dict[str, Any],
    density_profile: dict[str, Any],
    energy_abs_error: float,
    energy_tolerance: float,
    pure_r2_relative_tolerance: float,
    pure_rms_relative_tolerance: float,
    pure_density_l2_tolerance: float,
    density_accounting_tolerance: float,
    density_shape_min_bins: int,
) -> dict[str, Any]:
    exact_r2 = trapped_tg_r2_radius(anchor.n_particles, anchor.omega)
    exact_rms = trapped_tg_rms_radius(anchor.n_particles, anchor.omega)
    mixed_r2 = float(np.mean([summary.r2_radius for summary in seed_summaries]))
    mixed_rms = float(np.mean([summary.rms_radius for summary in seed_summaries]))
    pure_r2 = _optional_float(pure_summary.get("pure_r2"))
    pure_rms = _optional_float(pure_summary.get("rms_radius"))
    mixed_r2_rel = _relative_error(mixed_r2, exact_r2)
    mixed_rms_rel = _relative_error(mixed_rms, exact_rms)
    pure_r2_rel = _relative_error(pure_r2, exact_r2)
    pure_rms_rel = _relative_error(pure_rms, exact_rms)
    density_l2 = _density_relative_l2(
        density_profile.get("pure_fw_n_x"),
        density_profile.get("exact_bin_averaged_n_x"),
        density_profile.get("bin_edges"),
    )
    mixed_density_l2 = _density_relative_l2(
        density_profile.get("mixed_n_x"),
        density_profile.get("exact_bin_averaged_n_x"),
        density_profile.get("bin_edges"),
    )
    pure_mixed_density_l2 = _density_relative_l2(
        density_profile.get("pure_fw_n_x"),
        density_profile.get("mixed_n_x"),
        density_profile.get("bin_edges"),
    )
    pure_density_integral = _optional_float(density_profile.get("pure_fw_integral"))
    exact_density_integral = _optional_float(density_profile.get("exact_bin_averaged_integral"))
    density_accounting_abs_error = _optional_abs_difference(
        pure_density_integral,
        exact_density_integral,
    )
    density_bin_count = _density_bin_count(density_profile.get("bin_edges"))
    energy_accepted = energy_abs_error <= energy_tolerance
    r2_accepted = _passes_tolerance(pure_r2_rel, pure_r2_relative_tolerance)
    rms_accepted = _passes_tolerance(pure_rms_rel, pure_rms_relative_tolerance)
    density_l2_accepted = _passes_tolerance(density_l2, pure_density_l2_tolerance)
    density_resolution_accepted = density_bin_count >= density_shape_min_bins
    density_accounting_accepted = _passes_tolerance(
        density_accounting_abs_error,
        density_accounting_tolerance,
    )
    density_accepted = (
        density_l2_accepted and density_resolution_accepted and density_accounting_accepted
    )
    relative_errors = [
        value for value in (pure_r2_rel, pure_rms_rel, density_l2) if value is not None
    ]
    return {
        "status": _exact_comparison_status(
            energy_accepted=energy_accepted,
            pure_status=str(pure_summary.get("status", "not_evaluated")),
            r2_accepted=r2_accepted,
            rms_accepted=rms_accepted,
            density_l2_accepted=density_l2_accepted,
            density_resolution_accepted=density_resolution_accepted,
            density_accounting_accepted=density_accounting_accepted,
        ),
        "energy_status": "accepted" if energy_accepted else "reference_mismatch",
        "transported_fw_status": str(pure_summary.get("status", "not_evaluated")),
        "pure_r2_status": "accepted" if r2_accepted else "reference_mismatch",
        "pure_rms_status": "accepted" if rms_accepted else "reference_mismatch",
        "pure_density_status": "accepted" if density_accepted else "reference_mismatch",
        "pure_density_l2_status": ("accepted" if density_l2_accepted else "reference_mismatch"),
        "density_resolution_status": (
            "accepted" if density_resolution_accepted else "insufficient_resolution"
        ),
        "density_accounting_status": (
            "accepted" if density_accounting_accepted else "density_normalization_mismatch"
        ),
        "density_bin_count": density_bin_count,
        "density_shape_min_bins": density_shape_min_bins,
        "pure_density_integral": pure_density_integral,
        "exact_bin_averaged_density_integral": exact_density_integral,
        "density_accounting_abs_error": density_accounting_abs_error,
        "density_accounting_tolerance": density_accounting_tolerance,
        "exact_r2_radius": exact_r2,
        "mixed_r2_diagnostic": mixed_r2,
        "mixed_r2_relative_error_diagnostic": mixed_r2_rel,
        "pure_r2_radius": pure_r2,
        "pure_r2_relative_error": pure_r2_rel,
        "pure_r2_relative_tolerance": pure_r2_relative_tolerance,
        "exact_rms_radius": exact_rms,
        "mixed_rms_diagnostic": mixed_rms,
        "mixed_rms_relative_error_diagnostic": mixed_rms_rel,
        "pure_rms_radius": pure_rms,
        "pure_rms_relative_error": pure_rms_rel,
        "pure_rms_relative_tolerance": pure_rms_relative_tolerance,
        "pure_density_relative_l2": density_l2,
        "mixed_density_relative_l2_diagnostic": mixed_density_l2,
        "pure_mixed_density_relative_l2_diagnostic": pure_mixed_density_l2,
        "mixed_density_closer_than_fw_to_exact_diagnostic": (
            mixed_density_l2 is not None
            and density_l2 is not None
            and mixed_density_l2 < density_l2
        ),
        "pure_density_l2_tolerance": pure_density_l2_tolerance,
        "max_relative_observable_error": max(relative_errors) if relative_errors else None,
        "max_relative_tolerance": max(
            pure_r2_relative_tolerance,
            pure_rms_relative_tolerance,
            pure_density_l2_tolerance,
        ),
    }


def _exact_comparison_status(
    *,
    energy_accepted: bool,
    pure_status: str,
    r2_accepted: bool,
    rms_accepted: bool,
    density_l2_accepted: bool,
    density_resolution_accepted: bool,
    density_accounting_accepted: bool,
) -> str:
    if not energy_accepted:
        return "energy_reference_mismatch"
    if pure_status != "accepted":
        return pure_status
    if not r2_accepted:
        return "r2_reference_mismatch"
    if not rms_accepted:
        return "rms_reference_mismatch"
    if not density_l2_accepted:
        return "density_reference_mismatch"
    if not density_resolution_accepted:
        return "insufficient_density_resolution"
    if not density_accounting_accepted:
        return "density_normalization_mismatch"
    return "accepted"


def _density_bin_count(bin_edges: object) -> int:
    try:
        edges = np.asarray(bin_edges, dtype=float)
    except (TypeError, ValueError):
        return 0
    if edges.ndim != 1 or edges.size < 2:
        return 0
    return int(edges.size - 1)


def _bin_averaged_trapped_tg_density(
    bin_edges: np.ndarray,
    *,
    n_particles: int,
    omega: float,
) -> np.ndarray:
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must contain at least two points")
    values = np.empty(edges.size - 1, dtype=float)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        grid = np.linspace(float(left), float(right), 25)
        density = trapped_tg_density_profile(grid, n_particles=n_particles, omega=omega)
        values[index] = float(np.trapezoid(density, grid) / (right - left))
    return values


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real | str):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _optional_abs_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def _relative_error(value: float | None, exact: float) -> float | None:
    if value is None or exact == 0.0:
        return None
    return abs(value - exact) / abs(exact)


def _passes_tolerance(value: float | None, tolerance: float) -> bool:
    return value is not None and value <= tolerance


def _density_relative_l2(
    values: object,
    exact_values: object,
    bin_edges: object,
) -> float | None:
    if values is None:
        return None
    try:
        density = np.asarray(values, dtype=float)
        exact = np.asarray(exact_values, dtype=float)
        edges = np.asarray(bin_edges, dtype=float)
    except (TypeError, ValueError):
        return None
    if density.shape != exact.shape or edges.size != exact.size + 1:
        return None
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(exact)):
        return None
    widths = np.diff(edges)
    numerator = float(np.sum((density - exact) ** 2 * widths))
    denominator = float(np.sum(exact**2 * widths))
    if denominator <= 0.0:
        return None
    return float(np.sqrt(numerator / denominator))


def _density_stderr(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.full(values.shape[1], np.nan, dtype=float)
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _finite_list_or_none(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]
