from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.analysis import TraceStationarityResult, trace_stationarity_diagnostics
from hrdmc.estimators.mixed import WeightedObservableResult, estimate_weighted_observables

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WeightedDMCResult:
    """Generic weighted DMC sample contract.

    Concrete DMC engines may differ in transition kernels, population control,
    or reconfiguration, but downstream estimators consume this common shape.
    """

    snapshots: FloatArray
    local_energies: FloatArray
    weights: FloatArray
    metadata: dict

    def __post_init__(self) -> None:
        snapshots = np.asarray(self.snapshots, dtype=float)
        local_energies = np.asarray(self.local_energies, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if snapshots.ndim != 2:
            raise ValueError("snapshots must have shape (n_samples, n_particles)")
        if local_energies.shape != (snapshots.shape[0],):
            raise ValueError("local_energies must have one value per sample")
        if weights.shape != (snapshots.shape[0],):
            raise ValueError("weights must have one value per sample")

    def estimate_weighted_observables(
        self,
        *,
        valid_mask: NDArray[np.bool_],
        grid: FloatArray,
        center: float,
        n_particles: int,
    ) -> WeightedObservableResult:
        """Compute observables through the estimator owner."""

        return estimate_weighted_observables(
            self.snapshots,
            self.local_energies,
            self.weights,
            valid_mask,
            grid,
            center=center,
            n_particles=n_particles,
        )

    def trace_stationarity_diagnostics(
        self,
        times: FloatArray,
        values: FloatArray,
    ) -> TraceStationarityResult:
        """Audit a stored observable trace through the analysis owner."""

        return trace_stationarity_diagnostics(times, values)
