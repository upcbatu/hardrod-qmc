from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.estimators.density import DensityProfileResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WeightedConfigurationSet:
    samples: FloatArray
    local_energies: FloatArray
    normalized_weights: FloatArray
    candidate_sample_count: int
    included_sample_count: int
    raw_weight_sum: float
    effective_sample_size: float
    zero_weight_excluded_count: int
    invalid_excluded_count: int
    nonfinite_excluded_count: int
    finite_sample_fraction: float
    included_sample_fraction: float


@dataclass(frozen=True)
class WeightedObservableResult:
    mixed_energy: float
    rms_radius: float
    r2_radius: float
    density: DensityProfileResult
    effective_sample_size: float
    density_integral: float
    density_integral_abs_error: float
    lost_out_of_grid_sample_count: int
    lost_out_of_grid_weight: float
    edge_bin_occupancy: float


def filter_weighted_configurations(
    samples: FloatArray,
    local_energies: FloatArray,
    weights: FloatArray,
    valid_mask: NDArray[np.bool_],
) -> WeightedConfigurationSet:
    """Keep finite, valid, positive-weight configurations and normalize weights."""

    samples = np.asarray(samples, dtype=float)
    local_energies = np.asarray(local_energies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_particles)")
    if local_energies.shape != weights.shape or weights.shape != valid_mask.shape:
        raise ValueError("local_energies, weights, and valid_mask must have matching shapes")
    if weights.shape != (samples.shape[0],):
        raise ValueError("weights must have one value per sample")

    finite_mask = (
        np.all(np.isfinite(samples), axis=1)
        & np.isfinite(local_energies)
        & np.isfinite(weights)
    )
    positive_weight_mask = finite_mask & (weights > 0.0)
    included = positive_weight_mask & valid_mask
    raw_weight_sum = float(np.sum(weights[included]))
    if raw_weight_sum <= 0.0:
        raise ValueError("no finite positive-weight valid configurations remain")

    normalized_weights = weights[included] / raw_weight_sum
    return WeightedConfigurationSet(
        samples=samples[included],
        local_energies=local_energies[included],
        normalized_weights=normalized_weights,
        candidate_sample_count=int(weights.size),
        included_sample_count=int(np.count_nonzero(included)),
        raw_weight_sum=raw_weight_sum,
        effective_sample_size=float(1.0 / np.sum(normalized_weights * normalized_weights)),
        zero_weight_excluded_count=int(np.count_nonzero(finite_mask & (weights <= 0.0))),
        invalid_excluded_count=int(np.count_nonzero(positive_weight_mask & ~valid_mask)),
        nonfinite_excluded_count=int(np.count_nonzero(~finite_mask)),
        finite_sample_fraction=float(np.count_nonzero(finite_mask) / weights.size),
        included_sample_fraction=float(np.count_nonzero(included) / weights.size),
    )


def weighted_energy(local_energies: FloatArray, normalized_weights: FloatArray) -> float:
    local_energies = np.asarray(local_energies, dtype=float)
    normalized_weights = np.asarray(normalized_weights, dtype=float)
    if local_energies.shape != normalized_weights.shape:
        raise ValueError("local_energies and normalized_weights must have matching shapes")
    return float(np.sum(normalized_weights * local_energies))


def weighted_r2_radius(
    samples: FloatArray,
    normalized_weights: FloatArray,
    center: float = 0.0,
) -> float:
    samples = np.asarray(samples, dtype=float)
    normalized_weights = np.asarray(normalized_weights, dtype=float)
    per_configuration = np.mean((samples - center) ** 2, axis=1)
    return float(np.sum(normalized_weights * per_configuration))


def weighted_rms_radius(
    samples: FloatArray,
    normalized_weights: FloatArray,
    center: float = 0.0,
) -> float:
    return float(np.sqrt(weighted_r2_radius(samples, normalized_weights, center)))


def weighted_density_profile_on_grid(
    samples: FloatArray,
    normalized_weights: FloatArray,
    grid: FloatArray,
    n_particles: int,
) -> tuple[DensityProfileResult, dict[str, float | int]]:
    samples = np.asarray(samples, dtype=float)
    normalized_weights = np.asarray(normalized_weights, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_particles)")
    if normalized_weights.shape != (samples.shape[0],):
        raise ValueError("normalized_weights must have one value per sample")
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two points")
    if n_particles != samples.shape[1]:
        raise ValueError("n_particles must match samples.shape[1]")

    dx = float(grid[1] - grid[0])
    if dx <= 0.0 or not np.allclose(np.diff(grid), dx):
        raise ValueError("grid must be uniformly increasing")
    edges = np.concatenate(
        ([grid[0] - 0.5 * dx], 0.5 * (grid[:-1] + grid[1:]), [grid[-1] + 0.5 * dx])
    )
    flat_positions = samples.ravel()
    flat_weights = np.repeat(normalized_weights, samples.shape[1])
    bin_index = np.searchsorted(edges, flat_positions, side="right") - 1
    in_grid = (bin_index >= 0) & (bin_index < grid.size)
    hist = np.bincount(
        bin_index[in_grid],
        weights=flat_weights[in_grid],
        minlength=grid.size,
    ).astype(float)
    density = hist / dx
    density_integral = float(np.sum(density) * dx)
    diagnostics = {
        "density_integral": density_integral,
        "density_integral_abs_error": abs(density_integral - n_particles),
        "lost_out_of_grid_sample_count": int(np.count_nonzero(~in_grid)),
        "lost_out_of_grid_weight": float(np.sum(flat_weights[~in_grid])),
        "edge_bin_occupancy": float(hist[0] + hist[-1]),
    }
    return DensityProfileResult(x=grid, n_x=density, bin_edges=edges), diagnostics


def estimate_weighted_observables(
    samples: FloatArray,
    local_energies: FloatArray,
    weights: FloatArray,
    valid_mask: NDArray[np.bool_],
    grid: FloatArray,
    *,
    center: float,
    n_particles: int,
) -> WeightedObservableResult:
    filtered = filter_weighted_configurations(samples, local_energies, weights, valid_mask)
    density, density_diagnostics = weighted_density_profile_on_grid(
        filtered.samples,
        filtered.normalized_weights,
        grid,
        n_particles,
    )
    r2 = weighted_r2_radius(filtered.samples, filtered.normalized_weights, center)
    return WeightedObservableResult(
        mixed_energy=weighted_energy(filtered.local_energies, filtered.normalized_weights),
        rms_radius=float(np.sqrt(r2)),
        r2_radius=r2,
        density=density,
        effective_sample_size=filtered.effective_sample_size,
        density_integral=float(density_diagnostics["density_integral"]),
        density_integral_abs_error=float(density_diagnostics["density_integral_abs_error"]),
        lost_out_of_grid_sample_count=int(density_diagnostics["lost_out_of_grid_sample_count"]),
        lost_out_of_grid_weight=float(density_diagnostics["lost_out_of_grid_weight"]),
        edge_bin_occupancy=float(density_diagnostics["edge_bin_occupancy"]),
    )
