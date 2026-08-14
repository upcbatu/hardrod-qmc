from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShellPeriodAverage:
    """Density coarse-grained over interior unit-mass shell cells."""

    boundaries: FloatArray
    centers: FloatArray
    values: FloatArray
    replicate_centers: FloatArray | None
    replicate_values: FloatArray | None
    stderr: FloatArray | None


@dataclass(frozen=True)
class FixedShellCellAverage:
    """Density averaged over shell cells fixed by a reference profile."""

    boundaries: FloatArray
    centers: FloatArray
    values: FloatArray
    replicate_values: FloatArray | None


@dataclass(frozen=True)
class ShellPeakProfile:
    """Histogram peak positions and amplitudes inside fixed shell cells."""

    boundaries: FloatArray
    positions: FloatArray
    amplitudes: FloatArray


def _validated_histogram(
    bin_edges: FloatArray,
    density: FloatArray,
    *,
    name: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    edges = np.asarray(bin_edges, dtype=np.float64)
    values = np.asarray(density, dtype=np.float64)
    if edges.ndim != 1 or values.ndim != 1:
        raise ValueError(f"{name} histogram inputs must be one-dimensional")
    if values.size < 3 or edges.size != values.size + 1:
        raise ValueError(f"{name} bin_edges must contain one more value than density")
    if not np.all(np.isfinite(edges)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} histogram inputs must be finite")
    widths = np.diff(edges)
    if not np.all(widths > 0.0):
        raise ValueError(f"{name} bin_edges must be strictly increasing")
    if not np.all(values >= 0.0):
        raise ValueError(f"{name} density must be non-negative")
    return edges, values, widths


def _cumulative_edges(values: FloatArray, widths: FloatArray) -> FloatArray:
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values * widths)))


def _inverse_histogram_cumulative(
    edges: FloatArray,
    values: FloatArray,
    cumulative_edges: FloatArray,
    targets: FloatArray,
) -> FloatArray:
    positions = np.empty_like(targets, dtype=np.float64)
    for target_index, target in enumerate(targets):
        edge_index = int(np.searchsorted(cumulative_edges, target, side="left") - 1)
        if edge_index < 0 or edge_index >= values.size or values[edge_index] <= 0.0:
            raise ValueError("density cumulative is not invertible at a shell boundary")
        positions[target_index] = edges[edge_index] + (
            (target - cumulative_edges[edge_index]) / values[edge_index]
        )
    return positions


def _cumulative_at(
    points: FloatArray,
    edges: FloatArray,
    values: FloatArray,
    cumulative_edges: FloatArray,
) -> FloatArray:
    indices = np.searchsorted(edges, points, side="right") - 1
    indices = np.clip(indices, 0, values.size - 1)
    cumulative = cumulative_edges[indices] + values[indices] * (points - edges[indices])
    cumulative = np.where(points <= edges[0], 0.0, cumulative)
    cumulative = np.where(points >= edges[-1], cumulative_edges[-1], cumulative)
    return np.asarray(cumulative, dtype=np.float64)


def unit_mass_shell_average(
    bin_edges: FloatArray,
    density: FloatArray,
    *,
    particle_count: int,
    replicate_densities: FloatArray | None = None,
    normalization_atol: float = 1.0e-8,
) -> ShellPeriodAverage:
    """Average a histogram over adaptive cells containing one particle each."""
    if isinstance(particle_count, bool) or not isinstance(particle_count, (int, np.integer)):
        raise ValueError("particle_count must be an integer")
    if particle_count < 3:
        raise ValueError("particle_count must be at least three")
    if not np.isfinite(normalization_atol) or normalization_atol < 0.0:
        raise ValueError("normalization_atol must be finite and non-negative")
    edges, values, widths = _validated_histogram(bin_edges, density, name="aggregate")
    cumulative_edges = _cumulative_edges(values, widths)
    total_mass = float(cumulative_edges[-1])
    if not np.isclose(total_mass, particle_count, rtol=0.0, atol=normalization_atol):
        raise ValueError(
            f"aggregate density integrates to {total_mass:.12g}; expected {particle_count}"
        )
    mass_levels = np.arange(1, particle_count, dtype=np.float64)
    boundaries = _inverse_histogram_cumulative(
        edges,
        values,
        cumulative_edges,
        mass_levels,
    )
    cell_widths = np.diff(boundaries)
    if not np.all(cell_widths > 0.0):
        raise ValueError("unit-mass shell boundaries must be strictly increasing")
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    aggregate_masses = np.diff(_cumulative_at(boundaries, edges, values, cumulative_edges))
    if not np.allclose(aggregate_masses, 1.0, rtol=0.0, atol=5.0e-12):
        raise ValueError("unit-mass shell construction failed its mass-conservation check")
    shell_values = aggregate_masses / cell_widths
    replicate_values: FloatArray | None = None
    replicate_centers: FloatArray | None = None
    stderr: FloatArray | None = None
    if replicate_densities is not None:
        replicate_centers, replicate_values = _replicate_shell_averages(
            replicate_densities,
            edges=edges,
            widths=widths,
            mass_levels=mass_levels,
            particle_count=particle_count,
            normalization_atol=normalization_atol,
        )
        stderr = np.std(replicate_values, axis=0, ddof=1) / np.sqrt(replicate_values.shape[0])
    return ShellPeriodAverage(
        boundaries=np.asarray(boundaries, dtype=np.float64),
        centers=np.asarray(centers, dtype=np.float64),
        values=np.asarray(shell_values, dtype=np.float64),
        replicate_centers=replicate_centers,
        replicate_values=replicate_values,
        stderr=None if stderr is None else np.asarray(stderr, dtype=np.float64),
    )


def _replicate_shell_averages(
    replicate_densities: FloatArray,
    *,
    edges: FloatArray,
    widths: FloatArray,
    mass_levels: FloatArray,
    particle_count: int,
    normalization_atol: float,
) -> tuple[FloatArray, FloatArray]:
    replicates = np.asarray(replicate_densities, dtype=np.float64)
    if replicates.ndim != 2 or replicates.shape[1] != widths.size:
        raise ValueError("replicate_densities must have shape (replicates, bins)")
    if replicates.shape[0] < 2:
        raise ValueError("at least two replicate densities are required for a standard error")
    if not np.all(np.isfinite(replicates)) or np.any(replicates < 0.0):
        raise ValueError("replicate densities must be finite and non-negative")
    center_rows: list[FloatArray] = []
    value_rows: list[FloatArray] = []
    for index, replicate in enumerate(replicates):
        cumulative = _cumulative_edges(replicate, widths)
        mass = float(cumulative[-1])
        if not np.isclose(mass, particle_count, rtol=0.0, atol=normalization_atol):
            raise ValueError(
                f"replicate {index} integrates to {mass:.12g}; expected {particle_count}"
            )
        boundaries = _inverse_histogram_cumulative(edges, replicate, cumulative, mass_levels)
        replicate_widths = np.diff(boundaries)
        if not np.all(replicate_widths > 0.0):
            raise ValueError(f"replicate {index} shell boundaries are not increasing")
        center_rows.append(0.5 * (boundaries[:-1] + boundaries[1:]))
        value_rows.append(1.0 / replicate_widths)
    return np.asarray(center_rows, dtype=np.float64), np.asarray(value_rows, dtype=np.float64)


def fixed_shell_cell_average(
    bin_edges: FloatArray,
    density: FloatArray,
    *,
    boundaries: FloatArray,
    replicate_densities: FloatArray | None = None,
) -> FixedShellCellAverage:
    """Average densities over one common set of reference shell boundaries."""
    edges, values, widths = _validated_histogram(bin_edges, density, name="aggregate")
    cells = np.asarray(boundaries, dtype=np.float64)
    if cells.ndim != 1 or cells.size < 2 or not np.all(np.isfinite(cells)):
        raise ValueError("boundaries must contain at least two finite values")
    if not np.all(np.diff(cells) > 0.0):
        raise ValueError("boundaries must be strictly increasing")
    if cells[0] < edges[0] or cells[-1] > edges[-1]:
        raise ValueError("boundaries must lie inside the histogram grid")
    cumulative = _cumulative_edges(values, widths)
    cell_widths = np.diff(cells)
    cell_values = np.diff(_cumulative_at(cells, edges, values, cumulative)) / cell_widths
    replicate_values: FloatArray | None = None
    if replicate_densities is not None:
        replicates = np.asarray(replicate_densities, dtype=np.float64)
        if replicates.ndim != 2 or replicates.shape[1] != values.size:
            raise ValueError("replicate_densities must have shape (replicates, bins)")
        if not np.all(np.isfinite(replicates)) or not np.all(replicates >= 0.0):
            raise ValueError("replicate densities must be finite and non-negative")
        rows = []
        for replicate in replicates:
            replicate_cumulative = _cumulative_edges(replicate, widths)
            rows.append(
                np.diff(_cumulative_at(cells, edges, replicate, replicate_cumulative)) / cell_widths
            )
        replicate_values = np.asarray(rows, dtype=np.float64)
    return FixedShellCellAverage(
        boundaries=cells.copy(),
        centers=0.5 * (cells[:-1] + cells[1:]),
        values=np.asarray(cell_values, dtype=np.float64),
        replicate_values=replicate_values,
    )


def shell_peak_profile(
    bin_edges: FloatArray,
    density: FloatArray,
    *,
    boundaries: FloatArray,
) -> ShellPeakProfile:
    """Locate the largest histogram value inside each fixed shell cell."""
    edges, values, _ = _validated_histogram(bin_edges, density, name="aggregate")
    cells = np.asarray(boundaries, dtype=np.float64)
    if cells.ndim != 1 or cells.size < 2 or not np.all(np.isfinite(cells)):
        raise ValueError("boundaries must contain at least two finite values")
    if not np.all(np.diff(cells) > 0.0):
        raise ValueError("boundaries must be strictly increasing")
    if cells[0] < edges[0] or cells[-1] > edges[-1]:
        raise ValueError("boundaries must lie inside the histogram grid")
    positions = np.empty(cells.size - 1, dtype=np.float64)
    amplitudes = np.empty(cells.size - 1, dtype=np.float64)
    for cell_index, (left, right) in enumerate(itertools.pairwise(cells)):
        overlaps = (edges[:-1] < right) & (edges[1:] > left)
        candidate_indices = np.flatnonzero(overlaps)
        if candidate_indices.size == 0:
            raise ValueError("a fixed shell cell contains no histogram bins")
        local_index = int(candidate_indices[np.argmax(values[candidate_indices])])
        overlap_left = max(float(edges[local_index]), float(left))
        overlap_right = min(float(edges[local_index + 1]), float(right))
        positions[cell_index] = 0.5 * (overlap_left + overlap_right)
        amplitudes[cell_index] = values[local_index]
    return ShellPeakProfile(
        boundaries=cells.copy(),
        positions=positions,
        amplitudes=amplitudes,
    )


def relative_density_l2_error(
    x: FloatArray,
    estimate: FloatArray,
    reference: FloatArray,
) -> float:
    """Return ||estimate-reference||_2 / ||reference||_2 on a fixed grid."""
    grid = np.asarray(x, dtype=float)
    estimate_values = np.asarray(estimate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if grid.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if estimate_values.shape != grid.shape or reference_values.shape != grid.shape:
        raise ValueError("density arrays must match x shape")
    if grid.size < 2:
        raise ValueError("x must contain at least two points")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("x must be strictly increasing")
    numerator = float(np.trapezoid((estimate_values - reference_values) ** 2, grid))
    denominator = float(np.trapezoid(reference_values**2, grid))
    if denominator <= 0.0:
        raise ValueError("reference density norm must be positive")
    return float(np.sqrt(numerator / denominator))
