from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.hard_rods import HardRodSystem


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DensityProfileResult:
    x: FloatArray
    n_x: FloatArray
    bin_edges: FloatArray


def integrate_density_profile(profile: DensityProfileResult) -> float:
    """Integrate a histogram density profile using its bin widths."""
    widths = np.diff(profile.bin_edges)
    if widths.shape != profile.n_x.shape:
        raise ValueError("bin_edges must have one more entry than n_x")
    return float(np.sum(profile.n_x * widths))


def estimate_density_profile(
    snapshots: FloatArray,
    system: HardRodSystem,
    n_bins: int = 100,
) -> DensityProfileResult:
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_particles)")
    positions = system.wrap(snapshots.reshape(-1))
    counts, edges = np.histogram(positions, bins=n_bins, range=(0.0, system.length))
    dx = edges[1] - edges[0]
    n_x = counts / (snapshots.shape[0] * dx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return DensityProfileResult(x=centers, n_x=n_x, bin_edges=edges)


def estimate_open_line_density_profile(
    snapshots: FloatArray,
    x_min: float,
    x_max: float,
    n_bins: int = 100,
) -> DensityProfileResult:
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_particles)")
    if snapshots.shape[0] == 0:
        raise ValueError("at least one snapshot is required")
    if x_max <= x_min:
        raise ValueError("x_max must be larger than x_min")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    positions = snapshots.reshape(-1)
    counts, edges = np.histogram(positions, bins=n_bins, range=(x_min, x_max))
    dx = edges[1] - edges[0]
    n_x = counts / (snapshots.shape[0] * dx)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return DensityProfileResult(x=centers, n_x=n_x, bin_edges=edges)
