from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.hard_rods import HardRodSystem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PairDistributionResult:
    r: FloatArray
    g_r: FloatArray
    counts: FloatArray
    bin_edges: FloatArray


def estimate_pair_distribution(
    snapshots: FloatArray,
    system: HardRodSystem,
    n_bins: int = 100,
    r_max: float | None = None,
) -> PairDistributionResult:
    """Estimate pair distribution function g(r) from coordinate snapshots.

    Source/rationale
    ----------------
    g(r) is one of the hard-rod observables analyzed in
    [Mazzanti2008HardRods] to distinguish gas-like and solid-like regimes.

    Implementation convention
    -------------------------
    For unique pairs under minimum-image distances in one dimension, r in
    [0, L/2], the histogram is normalized so that approximately:

        expected_count_per_bin = n_samples * N * rho * dr * g(r)

    This normalization is the repository convention for the pair-correlation
    estimator; the physical observable is the same g(r) used in the hard-rod
    paper. For finite N, integrating the estimator over [0, L/2] gives the
    unique-pair sum rule L*(N-1)/(2*N).
    """
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_particles)")
    if snapshots.shape[1] != system.n_particles:
        raise ValueError("snapshot particle dimension does not match system")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    if r_max is None:
        r_max = system.length / 2.0
    if r_max <= 0 or r_max > system.length / 2.0 + 1e-12:
        raise ValueError("r_max must be in (0, L/2]")

    all_distances = []
    for positions in snapshots:
        all_distances.append(system.pair_distances_min_image(positions))
    distances = np.concatenate(all_distances) if all_distances else np.asarray([], dtype=float)

    counts, edges = np.histogram(distances, bins=n_bins, range=(0.0, r_max))
    dr = edges[1] - edges[0]
    expected = snapshots.shape[0] * system.n_particles * system.density * dr
    g_r = counts.astype(float) / expected if expected > 0 else np.zeros_like(counts, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return PairDistributionResult(r=centers, g_r=g_r, counts=counts.astype(float), bin_edges=edges)
