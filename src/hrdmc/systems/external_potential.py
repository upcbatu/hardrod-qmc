from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BASE_TRAP_QUADRATIC_COUPLING = 0.5


def lambda_from_relative_offset(
    relative_offset: float,
    *,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
) -> float:
    """Map a relative trap-coupling offset to ``lambda=lambda0*(1+epsilon)``.

    The trapped oscillator-unit Hamiltonian has ``lambda0=1/2`` in
    ``V(q)=lambda0*(q-q0)^2``.  Keeping this mapping with the external
    potential gives guides, estimators, and workflows one Hamiltonian owner.
    """

    if not np.isfinite(relative_offset):
        raise ValueError("relative_offset must be finite")
    if not np.isfinite(lambda0) or not np.isclose(
        lambda0,
        BASE_TRAP_QUADRATIC_COUPLING,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("lambda0 must equal the oscillator-unit base coupling 0.5")
    lambda_value = float(lambda0 * (1.0 + relative_offset))
    if lambda_value <= 0.0:
        raise ValueError("relative_offset must keep lambda positive")
    return lambda_value


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
class HarmonicTrap:
    """Harmonic trap V(x)=0.5*omega^2*(x-center)^2.

    Trapped workflows use dimensionless harmonic-oscillator coordinates by
    default. In those code variables the default trap coefficient is one, so
    V(q)=q^2/2.
    """

    omega: float
    center: float = 0.0

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive")

    def values(self, positions: FloatArray) -> FloatArray:
        x = np.asarray(positions, dtype=float)
        return 0.5 * self.omega**2 * (x - self.center) ** 2

    def total(self, positions: FloatArray) -> float:
        return float(np.sum(self.values(positions)))


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
