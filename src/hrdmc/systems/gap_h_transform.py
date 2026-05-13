from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics import backend_label
from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.gap_h_pair_table import GapHTransformTable, build_gap_h_transform_table
from hrdmc.systems.harmonic_com_transition import (
    harmonic_com_ground_variance,
    harmonic_com_h_transform_log_density,
    sample_harmonic_com_h_transform,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.systems.open_line_gap_coordinates import (
    as_position_batch,
    positions_from_center_and_gaps,
    validate_particle_shape,
)

FloatArray = NDArray[np.float64]


@dataclass
class OpenHardRodTrapGapHTransformProposalKernel:
    """Gap-space h-transform proposal for trapped open-line RN-DMC.

    Systems owns this proposal because it is purely coordinate geometry plus
    system Green-kernel structure. The RN engine still owns sampling weights,
    population control, and acceptance accounting.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    pair_grid_points: int = 700
    pair_y_max: float | None = None
    _tables: dict[float, GapHTransformTable] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")
        if self.system.n_particles < 2:
            raise ValueError("gap h-transform proposal requires at least two particles")

    def sample(
        self,
        rng: np.random.Generator,
        x_old: FloatArray,
        tau: float,
    ) -> FloatArray:
        x_old = as_position_batch(x_old)
        validate_particle_shape(x_old, self.system.n_particles)
        q_old = np.mean(x_old - self.system.center, axis=1)
        gaps_old = np.diff(x_old, axis=1)
        table = self._table(tau)
        q_new = sample_harmonic_com_h_transform(
            rng,
            q_old,
            n_particles=self.system.n_particles,
            omega=self.trap.omega,
            tau=tau,
        )
        gaps_new = table.sample_gaps(rng, gaps_old)
        return positions_from_center_and_gaps(
            q_new,
            gaps_new,
            center=self.system.center,
        )

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = as_position_batch(x_old)
        x_new = as_position_batch(x_new)
        validate_particle_shape(x_old, self.system.n_particles)
        validate_particle_shape(x_new, self.system.n_particles)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        q_old = np.mean(x_old - self.system.center, axis=1)
        q_new = np.mean(x_new - self.system.center, axis=1)
        gaps_old = np.diff(x_old, axis=1)
        gaps_new = np.diff(x_new, axis=1)
        log_com = harmonic_com_h_transform_log_density(
            q_old,
            q_new,
            n_particles=self.system.n_particles,
            omega=self.trap.omega,
            tau=tau,
        )
        return log_com + self._table(tau).log_density(gaps_old, gaps_new)

    @property
    def transition_backend(self) -> str:
        return backend_label("gap_h_transform")

    def _table(self, tau: float) -> GapHTransformTable:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        key = float(tau)
        table = self._tables.get(key)
        if table is None:
            table = build_gap_h_transform_table(
                rod_length=self.system.rod_length,
                omega=self.trap.omega,
                tau=key,
                grid_points=self.pair_grid_points,
                y_max=self.pair_y_max,
            )
            self._tables[key] = table
        return table


@dataclass
class OpenHardRodTrapGapHProductTargetKernel:
    """Approximate raw finite-a target from independent exact gap h-transforms.

    This target uses the same center-of-mass h-transform and exact N=2 gap
    h-transform tables as the gap-h proposal, but converts the product
    h-transform back to a raw kernel:

        K_product = p_h_product * exp(-tau * E_product)
                    * psi_product(old) / psi_product(new).

    For N=2 this reduces to the deterministic relative-coordinate exact target.
    For N>2 it is the production candidate that replaces the primitive target
    with an exact-two-body gap product approximation.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    pair_grid_points: int = 700
    pair_y_max: float | None = None
    _tables: dict[float, GapHTransformTable] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")
        if self.system.n_particles < 2:
            raise ValueError("gap-h product target requires at least two particles")
        if self.system.rod_length <= 0.0:
            raise ValueError("gap-h product target requires positive rod length")

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = as_position_batch(x_old)
        x_new = as_position_batch(x_new)
        validate_particle_shape(x_old, self.system.n_particles)
        validate_particle_shape(x_new, self.system.n_particles)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        q_old = np.mean(x_old - self.system.center, axis=1)
        q_new = np.mean(x_new - self.system.center, axis=1)
        gaps_old = np.diff(x_old, axis=1)
        gaps_new = np.diff(x_new, axis=1)
        table = self._table(tau)
        log_h = harmonic_com_h_transform_log_density(
            q_old,
            q_new,
            n_particles=self.system.n_particles,
            omega=self.trap.omega,
            tau=tau,
        )
        log_h += table.log_density(gaps_old, gaps_new)
        log_psi_old = self._ground_log_value(q_old, gaps_old, table)
        log_psi_new = self._ground_log_value(q_new, gaps_new, table)
        product_energy = (
            self.trap.omega / math.sqrt(2.0)
            + (self.system.n_particles - 1) * table.relative_energy
        )
        out = log_h - tau * product_energy + log_psi_old - log_psi_new
        invalid = (
            np.any(~np.isfinite(x_old), axis=1)
            | np.any(~np.isfinite(x_new), axis=1)
            | np.any(gaps_old <= self.system.rod_length, axis=1)
            | np.any(gaps_new <= self.system.rod_length, axis=1)
            | np.any(gaps_old > table.y_grid[-1], axis=1)
            | np.any(gaps_new > table.y_grid[-1], axis=1)
        )
        return np.where(invalid, -np.inf, out)

    @property
    def transition_backend(self) -> str:
        return "gap_h_product_target_numpy"

    def _table(self, tau: float) -> GapHTransformTable:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        key = float(tau)
        table = self._tables.get(key)
        if table is None:
            table = build_gap_h_transform_table(
                rod_length=self.system.rod_length,
                omega=self.trap.omega,
                tau=key,
                grid_points=self.pair_grid_points,
                y_max=self.pair_y_max,
            )
            self._tables[key] = table
        return table

    def _ground_log_value(
        self,
        q: FloatArray,
        gaps: FloatArray,
        table: GapHTransformTable,
    ) -> FloatArray:
        variance = harmonic_com_ground_variance(self.system.n_particles, self.trap.omega)
        log_com = -(q * q) / (4.0 * variance)
        psi_floor = max(
            float(np.max(np.abs(table.ground_state))) * 1.0e-14,
            np.finfo(float).tiny,
        )
        log_grid = np.log(np.maximum(table.ground_state, psi_floor))
        flat = gaps.reshape(-1)
        log_relative = np.interp(
            flat,
            table.y_grid,
            log_grid,
            left=-np.inf,
            right=-np.inf,
        ).reshape(gaps.shape)
        return log_com + np.sum(log_relative, axis=1)


@dataclass
class OpenN2HardRodTrapExactKernel:
    """Raw N=2 finite-a trapped hard-rod heat kernel from the relative solver.

    The table stores the normalized ground-state h-transform for the relative
    gap. RN-DMC requires the raw Hamiltonian kernel, so this target converts the
    h-transform density back with

        K = p_h * exp(-tau * E0) * psi0(old) / psi0(new).

    With the matching gap-h guide and the same h-transform proposal, the guide
    ratio in the RN increment cancels the psi0 ratio and leaves only the
    expected constant energy shift when the proposal exactly equals p_h.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    pair_grid_points: int = 700
    pair_y_max: float | None = None
    _tables: dict[float, GapHTransformTable] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")
        if self.system.n_particles != 2:
            raise ValueError("N=2 exact relative target requires exactly two particles")
        if self.system.rod_length <= 0.0:
            raise ValueError("N=2 exact relative target requires positive rod length")

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = as_position_batch(x_old)
        x_new = as_position_batch(x_new)
        validate_particle_shape(x_old, self.system.n_particles)
        validate_particle_shape(x_new, self.system.n_particles)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        q_old = np.mean(x_old - self.system.center, axis=1)
        q_new = np.mean(x_new - self.system.center, axis=1)
        gap_old = np.diff(x_old, axis=1)
        gap_new = np.diff(x_new, axis=1)
        table = self._table(tau)
        log_h = harmonic_com_h_transform_log_density(
            q_old,
            q_new,
            n_particles=self.system.n_particles,
            omega=self.trap.omega,
            tau=tau,
        )
        log_h += table.log_density(gap_old, gap_new)
        log_psi_old = self._ground_log_value(q_old, gap_old[:, 0], table)
        log_psi_new = self._ground_log_value(q_new, gap_new[:, 0], table)
        total_energy = table.relative_energy + self.trap.omega / math.sqrt(2.0)
        out = log_h - tau * total_energy + log_psi_old - log_psi_new
        invalid = (
            np.any(~np.isfinite(x_old), axis=1)
            | np.any(~np.isfinite(x_new), axis=1)
            | (gap_old[:, 0] <= self.system.rod_length)
            | (gap_new[:, 0] <= self.system.rod_length)
            | (gap_old[:, 0] > table.y_grid[-1])
            | (gap_new[:, 0] > table.y_grid[-1])
        )
        return np.where(invalid, -np.inf, out)

    @property
    def transition_backend(self) -> str:
        return "n2_exact_relative_numpy"

    def _table(self, tau: float) -> GapHTransformTable:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        key = float(tau)
        table = self._tables.get(key)
        if table is None:
            table = build_gap_h_transform_table(
                rod_length=self.system.rod_length,
                omega=self.trap.omega,
                tau=key,
                grid_points=self.pair_grid_points,
                y_max=self.pair_y_max,
            )
            self._tables[key] = table
        return table

    def _ground_log_value(
        self,
        q: FloatArray,
        gap: FloatArray,
        table: GapHTransformTable,
    ) -> FloatArray:
        variance = 1.0 / (math.sqrt(2.0) * self.system.n_particles * self.trap.omega)
        log_com = -(q * q) / (4.0 * variance)
        psi_floor = max(
            float(np.max(np.abs(table.ground_state))) * 1.0e-14,
            np.finfo(float).tiny,
        )
        log_relative = np.interp(
            gap,
            table.y_grid,
            np.log(np.maximum(table.ground_state, psi_floor)),
            left=-np.inf,
            right=-np.inf,
        )
        return log_com + log_relative
