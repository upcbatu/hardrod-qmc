from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.common.population import (
    normalize_weights as _normalize_weights,
)
from hrdmc.monte_carlo.dmc.common.population import (
    systematic_resample as _systematic_resample,
)
from hrdmc.monte_carlo.dmc.common.results import WeightedDMCResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DMCResult = WeightedDMCResult


@dataclass
class WalkerSet:
    """Container for a DMC walker population.

    Shape convention: positions has shape `(n_walkers, n_particles)`.
    """

    positions: FloatArray
    weights: FloatArray | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape (n_walkers, n_particles)")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != (self.positions.shape[0],):
                raise ValueError("weights must have shape (n_walkers,)")

    @property
    def n_walkers(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[1])


def normalize_weights(weights: FloatArray) -> FloatArray:
    return _normalize_weights(weights)


def systematic_resample(weights: FloatArray, rng: np.random.Generator) -> IntArray:
    return _systematic_resample(weights, rng)


class DiffusionMonteCarloEngine(Protocol):
    """Protocol seam shared by concrete DMC engines.

    Engines may implement different transition kernels or population control,
    but downstream analysis should consume the common weighted DMC result.
    """

    def run(self, *args: Any, **kwargs: Any) -> DMCResult: ...
