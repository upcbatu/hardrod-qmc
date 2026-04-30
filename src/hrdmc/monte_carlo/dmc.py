from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DMCResult:
    """Target data contract for the thesis-level DMC implementation.

    Source/rationale
    ----------------
    The eventual engine should support DMC mixed estimators plus forward-walking
    pure estimators. The pure-estimator methodology is from
    [BoronatCasulleras1995PureEstimators] and [SarsaBoronatCasulleras2002QuadraticDMC].
    The physical hard-rod benchmark is [Mazzanti2008HardRods].
    """

    snapshots: FloatArray
    local_energies: FloatArray
    weights: FloatArray
    ancestry: FloatArray | None = None
    metadata: dict | None = None


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
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return weights / total


def systematic_resample(weights: FloatArray, rng: np.random.Generator) -> IntArray:
    """Systematic resampling indices for future DMC population control."""
    w = normalize_weights(weights)
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(w)
    return np.searchsorted(cumulative, positions).astype(np.int64)


@dataclass
class AncestryBuffer:
    """Rolling ancestry buffer for forward-walking pure estimators.

    Source/rationale
    ----------------
    The pure-estimator plan is based on forward walking as developed for QMC
    expectation values by Casulleras and Boronat [BoronatCasulleras1995PureEstimators]
    and used in the quadratic DMC / pure-estimator implementation of Sarsa,
    Boronat and Casulleras [SarsaBoronatCasulleras2002QuadraticDMC].

    This object is only a data-structure seam. The final DMC engine should push
    ancestry index arrays into this object at every branching step so that
    descendant weights can be attached to older walker observables.
    """

    max_lag: int
    _history: list[IntArray] = field(default_factory=list)

    def push(self, parent_indices: IntArray) -> None:
        parent_indices = np.asarray(parent_indices, dtype=np.int64)
        self._history.append(parent_indices)
        if len(self._history) > self.max_lag:
            self._history.pop(0)

    def ready(self) -> bool:
        return len(self._history) == self.max_lag

    def compose_ancestry(self) -> IntArray:
        if not self._history:
            raise ValueError("ancestry history is empty")
        ancestry = np.arange(len(self._history[-1]), dtype=np.int64)
        for parents in reversed(self._history):
            ancestry = parents[ancestry]
        return ancestry


class DiffusionMonteCarloEngine:
    """Explicit seam for the final DMC engine.

    This scaffold avoids mixing DMC implementation details with observables and
    analysis. A self-written DMC engine or group-provided code should emit a
    `DMCResult` so the downstream benchmark stack remains unchanged.
    """

    def run(self, *args, **kwargs) -> DMCResult:  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "DMC production engine is not implemented in the bootstrap scaffold. "
            "Implement or adapt a DMC engine behind this interface."
        )
