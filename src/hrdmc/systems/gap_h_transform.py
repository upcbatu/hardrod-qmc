from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics import backend_label
from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.gap_h_pair_table import GapHTransformTable, build_gap_h_transform_table
from hrdmc.systems.harmonic_com_transition import (
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
