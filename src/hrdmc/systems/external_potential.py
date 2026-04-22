from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class ExternalPotential(Protocol):
    def values(self, positions: FloatArray) -> FloatArray: ...

    def total(self, positions: FloatArray) -> float: ...


@dataclass(frozen=True)
class ZeroPotential:
    def values(self, positions: FloatArray) -> FloatArray:
        return np.zeros_like(np.asarray(positions, dtype=float))

    def total(self, positions: FloatArray) -> float:
        return 0.0


@dataclass(frozen=True)
class CosinePotential:
    """Weak periodic external potential V0 cos(2 pi x / wavelength)."""

    amplitude: float
    wavelength: float

    def values(self, positions: FloatArray) -> FloatArray:
        x = np.asarray(positions, dtype=float)
        return self.amplitude * np.cos(2.0 * np.pi * x / self.wavelength)

    def total(self, positions: FloatArray) -> float:
        return float(np.sum(self.values(positions)))
