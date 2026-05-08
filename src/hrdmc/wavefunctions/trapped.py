from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.open_line import OpenLineHardRodSystem

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrappedHardRodTrial:
    """Diagnostic trapped hard-rod trial state.

    The form combines a one-body Gaussian trap factor with a nearest-neighbor
    hard-rod contact factor. It is a VMC diagnostic trial state, not a claimed
    exact trapped hard-rod wavefunction.
    """

    system: OpenLineHardRodSystem
    gaussian_alpha: float
    contact_power: float = 1.0

    def __post_init__(self) -> None:
        if self.gaussian_alpha <= 0:
            raise ValueError("gaussian_alpha must be positive")
        if self.contact_power < 0:
            raise ValueError("contact_power must be non-negative")

    def log_value(self, positions: FloatArray) -> float:
        positions = np.asarray(positions, dtype=float)
        if not self.system.is_valid(positions):
            return float("-inf")

        shifted = positions - self.system.center
        logv = -0.5 * self.gaussian_alpha * float(np.sum(shifted**2))
        if self.contact_power == 0:
            return logv

        gaps = self.system.nearest_neighbor_gaps(positions)
        free_gaps = gaps - self.system.rod_length
        if np.any(free_gaps <= 0):
            return float("-inf")
        return float(logv + self.contact_power * np.sum(np.log(free_gaps)))

    def value(self, positions: FloatArray) -> float:
        logv = self.log_value(positions)
        if not np.isfinite(logv):
            return 0.0
        return float(np.exp(logv))
