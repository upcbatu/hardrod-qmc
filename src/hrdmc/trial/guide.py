from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.system.geometry import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.trial.kernel import (
    backend_name,
    reduced_tg_grad_lap_local_batch,
    reduced_tg_log_batch,
    reduced_tg_relative_width_local_energy_batch,
    valid_batch,
)

if TYPE_CHECKING:
    from hrdmc.system.settings import TrappedCase

GUIDE_FAMILIES = ("reduced-tg",)
DEFAULT_GUIDE_FAMILY = "reduced-tg"

FloatArray = NDArray[np.float64]
class DMCGuide(Protocol):
    """Importance-sampling guide required by production DMC engines."""
    def log_value(self, positions: FloatArray) -> float: ...
    def grad_log_value(self, positions: FloatArray) -> FloatArray: ...
    def lap_log_value(self, positions: FloatArray) -> FloatArray: ...
    def local_energy(self, positions: FloatArray) -> float: ...
    def is_valid(self, positions: FloatArray) -> bool: ...
class BatchedDMCGuide(DMCGuide, Protocol):
    """Optional vectorized guide interface used by high-throughput DMC runs."""
    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]: ...
    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]: ...
    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]: ...
@dataclass(frozen=True)
class ReducedTGHardRodGuide:
    """DMC guide for trapped hard rods in reduced coordinates."""
    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    alpha: float
    relative_alpha: float | None = None
    pair_power: float = 1.0
    def __post_init__(self) -> None:
        if self.trap.center != self.system.center:
            raise ValueError("system and trap centers must match")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.relative_alpha is not None and self.relative_alpha <= 0.0:
            raise ValueError("relative_alpha must be positive when provided")
        if self.pair_power <= 0.0:
            raise ValueError("pair_power must be positive")
    def is_valid(self, positions: FloatArray) -> bool:
        positions = np.asarray(positions, dtype=float)
        return bool(
            positions.shape == (self.system.n_particles,)
            and np.all(np.isfinite(positions))
            and np.all(np.diff(positions) >= self.system.rod_length)
        )
    def log_value(self, positions: FloatArray) -> float:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.system.n_particles,):
            return float("-inf")
        row = positions[np.newaxis, :]
        log_values, _finite = self.batch_log_value(row)
        return float(log_values[0])
    def grad_log_value(self, positions: FloatArray) -> FloatArray:
        grad, _lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return grad[0]
    def lap_log_value(self, positions: FloatArray) -> FloatArray:
        _grad, lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return lap[0]
    def local_energy(self, positions: FloatArray) -> float:
        _grad, _lap, local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return float(local[0])
    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]:
        positions = self._as_position_batch(positions)
        return valid_batch(positions, self.system.rod_length)
    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
        positions = self._as_position_batch(positions)
        return reduced_tg_log_batch(
            positions,
            self._offsets(),
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            relative_alpha=self._relative_alpha(),
            center=self.system.center,
            pair_power=self.pair_power,
        )
    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        positions = self._as_position_batch(positions)
        grad, lap, local, finite = reduced_tg_grad_lap_local_batch(
            positions,
            self._offsets(),
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            relative_alpha=self._relative_alpha(),
            center=self.system.center,
            omega2=self.trap.omega**2,
            pair_power=self.pair_power,
        )
        if self.pair_power == 1.0 and np.isclose(self.alpha, self.trap.omega):
            closed_local = reduced_tg_relative_width_local_energy_batch(
                positions,
                rod_length=self.system.rod_length,
                omega=self.trap.omega,
                relative_alpha=self._relative_alpha(),
            )
            local = np.where(finite, closed_local, local)
        return grad, lap, local, finite
    @property
    def batch_backend(self) -> str:
        return backend_name()
    def _offsets(self) -> FloatArray:
        return self.system.rod_length * (
            np.arange(self.system.n_particles, dtype=float) - 0.5 * (self.system.n_particles - 1)
        )
    def _relative_alpha(self) -> float:
        return self.alpha if self.relative_alpha is None else self.relative_alpha
    def _as_position_batch(self, positions: FloatArray) -> FloatArray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != self.system.n_particles:
            raise ValueError("positions must have shape (n_walkers, n_particles)")
        return positions

def build_guide(
    case: TrappedCase,
    system: OpenLineHardRodSystem,
    trap: HarmonicTrap,
    *,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    relative_alpha: float | None = None,
) -> ReducedTGHardRodGuide:
    """Build only the importance-sampling guide for a trapped DMC run."""
    if guide_family not in GUIDE_FAMILIES:
        raise ValueError(f"unknown guide family: {guide_family}")
    return ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=case.omega,
        relative_alpha=relative_alpha,
    )
