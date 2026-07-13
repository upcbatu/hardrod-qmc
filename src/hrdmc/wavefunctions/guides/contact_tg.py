from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.guides.contact_correction import (
    N2GapContactCorrection,
    build_n2_gap_contact_correction,
)
from hrdmc.wavefunctions.guides.trapped_tg import ReducedTGHardRodGuide
from hrdmc.wavefunctions.kernels.contact_tg import (
    ContactTGSufficientStatistics,
    contact_tg_backend_name,
    contact_tg_grad_lap_local_batch,
    contact_tg_log_batch,
    contact_tg_sufficient_statistics_batch,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ContactCorrectedReducedTGHardRodGuide:
    """Reduced-TG DMC guide with a partial contact-regular N=2 correction.

    ``contact_beta=0`` dispatches directly to ``ReducedTGHardRodGuide``.
    Positive beta values interpolate toward the exact N=2 nearest-gap shape
    without changing the linear hard-core node.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    alpha: float
    relative_alpha: float | None = None
    contact_beta: float = 0.0
    pair_power: float = 1.0
    pair_grid_points: int = 700
    pair_y_max: float | None = None
    _base: ReducedTGHardRodGuide = field(init=False, repr=False)
    _correction: N2GapContactCorrection = field(init=False, repr=False)
    _offset_values: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")
        if not np.isclose(self.alpha, self.trap.omega, rtol=0.0, atol=1.0e-12):
            raise ValueError("contact-corrected reduced-TG guide requires alpha=trap omega")
        if self.pair_power != 1.0:
            raise ValueError("contact-corrected reduced-TG guide requires pair_power=1")
        if not np.isfinite(self.contact_beta) or not 0.0 <= self.contact_beta <= 1.0:
            raise ValueError("contact_beta must lie in [0, 1]")
        base = ReducedTGHardRodGuide(
            system=self.system,
            trap=self.trap,
            alpha=self.alpha,
            relative_alpha=self.relative_alpha,
            pair_power=self.pair_power,
        )
        correction = build_n2_gap_contact_correction(
            rod_length=self.system.rod_length,
            omega=self.trap.omega,
            grid_points=self.pair_grid_points,
            y_max=self.pair_y_max,
        )
        offsets = self.system.rod_length * (
            np.arange(self.system.n_particles, dtype=float) - 0.5 * (self.system.n_particles - 1)
        )
        offsets.setflags(write=False)
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_correction", correction)
        object.__setattr__(self, "_offset_values", offsets)

    @property
    def correction(self) -> N2GapContactCorrection:
        return self._correction

    def is_valid(self, positions: FloatArray) -> bool:
        return self._base.is_valid(positions)

    def log_value(self, positions: FloatArray) -> float:
        if self.contact_beta == 0.0:
            return self._base.log_value(positions)
        row = np.asarray(positions, dtype=float)
        if row.shape != (self.system.n_particles,):
            return float("-inf")
        values, finite = self.batch_log_value(row[np.newaxis, :])
        return float(values[0]) if finite[0] else float("-inf")

    def grad_log_value(self, positions: FloatArray) -> FloatArray:
        if self.contact_beta == 0.0:
            return self._base.grad_log_value(positions)
        grad, _lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod contact guide configuration")
        return grad[0]

    def lap_log_value(self, positions: FloatArray) -> FloatArray:
        if self.contact_beta == 0.0:
            return self._base.lap_log_value(positions)
        _grad, lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod contact guide configuration")
        return lap[0]

    def local_energy(self, positions: FloatArray) -> float:
        if self.contact_beta == 0.0:
            return self._base.local_energy(positions)
        _grad, _lap, local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod contact guide configuration")
        return float(local[0])

    def reduced_positions(self, positions: FloatArray) -> FloatArray:
        return np.asarray(positions, dtype=float) - self._offset_values

    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]:
        return self._base.valid_batch(self._as_position_batch(positions))

    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
        values = self._as_position_batch(positions)
        if self.contact_beta == 0.0:
            return self._base.batch_log_value(values)
        correction = self._correction
        return contact_tg_log_batch(
            values,
            self._offset_values,
            correction.breakpoints,
            correction.coefficients,
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            relative_alpha=self._relative_alpha(),
            center=self.system.center,
            pair_power=self.pair_power,
            contact_beta=self.contact_beta,
            omega=self.trap.omega,
            tail_nu=correction.tail_nu,
            tail_constant=correction.tail_constant,
            zero_correction=correction.zero_correction,
        )

    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        values = self._as_position_batch(positions)
        if self.contact_beta == 0.0:
            return self._base.batch_grad_lap_local(values)
        correction = self._correction
        return contact_tg_grad_lap_local_batch(
            values,
            self._offset_values,
            correction.breakpoints,
            correction.coefficients,
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            relative_alpha=self._relative_alpha(),
            center=self.system.center,
            omega=self.trap.omega,
            pair_power=self.pair_power,
            contact_beta=self.contact_beta,
            tail_nu=correction.tail_nu,
            tail_constant=correction.tail_constant,
            zero_correction=correction.zero_correction,
        )

    def sufficient_statistics_batch(
        self,
        positions: FloatArray,
    ) -> ContactTGSufficientStatistics:
        correction = self._correction
        return contact_tg_sufficient_statistics_batch(
            self._as_position_batch(positions),
            self._offset_values,
            correction.breakpoints,
            correction.coefficients,
            rod_length=self.system.rod_length,
            center=self.system.center,
            omega=self.trap.omega,
            tail_nu=correction.tail_nu,
            tail_constant=correction.tail_constant,
            zero_correction=correction.zero_correction,
        )

    @property
    def batch_backend(self) -> str:
        if self.contact_beta == 0.0:
            return self._base.batch_backend
        return f"contact_tg_{contact_tg_backend_name()}"

    def _relative_alpha(self) -> float:
        return self.alpha if self.relative_alpha is None else self.relative_alpha

    def _as_position_batch(self, positions: FloatArray) -> FloatArray:
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.system.n_particles:
            raise ValueError("positions must have shape (n_walkers, n_particles)")
        return values
