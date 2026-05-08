from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import excluded_length

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HardRodJastrowTrial:
    """Jastrow-like hard-rod trial wavefunction.

    Source/rationale
    ----------------
    The all-pair reduced-coordinate variant follows the analytic hard-rod
    ground-state structure used as the benchmark in [Mazzanti2008HardRods]:
    within one ordered sector, hard rods are mapped to point-like coordinates
    on the reduced length L' = L - N a.

    The nearest-neighbor-only mode is NOT claimed as a paper equation. It is a
    lightweight scaffold trial amplitude for early VMC smoke tests. Final DMC
    benchmarks should use the all-pair form or supervisor-provided trial forms.
    """

    system: HardRodSystem
    power: float = 1.0
    nearest_neighbor_only: bool = True

    def log_value(self, positions: FloatArray) -> float:
        if not self.system.is_valid(positions):
            return float("-inf")

        if self.nearest_neighbor_only:
            return self._log_nearest_neighbor(positions)
        return self._log_all_pairs_reduced(positions)

    def value(self, positions: FloatArray) -> float:
        logv = self.log_value(positions)
        if not np.isfinite(logv):
            return 0.0
        return float(np.exp(logv))

    def local_kinetic_energy(self, positions: FloatArray) -> float:
        """Return -sum_i nabla_i^2 psi / psi for the all-pair reduced trial.

        The all-pair `power=1` form is the controlled homogeneous-ring
        validation wavefunction. For that case this local kinetic energy is
        constant and equals the finite-N hard-rod benchmark energy.
        """

        if self.nearest_neighbor_only:
            raise NotImplementedError("local kinetic energy is only implemented for all-pair trial")
        if not self.system.is_valid(positions):
            return float("inf")

        x = self.system.sorted_positions(positions)
        n_particles = self.system.n_particles
        free_length = excluded_length(n_particles, self.system.length, self.system.rod_length)
        alpha = np.pi / free_length
        y = x - self.system.rod_length * np.arange(n_particles)

        laplacian_over_value = 0.0
        for i in range(n_particles):
            grad_log = 0.0
            lap_log = 0.0
            for j in range(n_particles):
                if i == j:
                    continue
                phase = alpha * (y[i] - y[j])
                sin_phase = np.sin(phase)
                if sin_phase == 0.0:
                    return float("inf")
                grad_log += self.power * alpha / np.tan(phase)
                lap_log -= self.power * alpha**2 / sin_phase**2
            laplacian_over_value += lap_log + grad_log**2

        return float(-laplacian_over_value)

    def _log_nearest_neighbor(self, positions: FloatArray) -> float:
        gaps = self.system.nearest_neighbor_gaps(positions)
        free_gaps = gaps - self.system.rod_length
        if np.any(free_gaps <= 0):
            return float("-inf")
        free_length = excluded_length(
            self.system.n_particles,
            self.system.length,
            self.system.rod_length,
        )
        # Prototype-only smooth positive amplitude on the simplex of free gaps.
        # This is an engineering scaffold, not a cited hard-rod equation.
        s = np.sin(np.pi * free_gaps / free_length)
        if np.any(s <= 0):
            return float("-inf")
        return float(self.power * np.sum(np.log(s)))

    def _log_all_pairs_reduced(self, positions: FloatArray) -> float:
        x = self.system.sorted_positions(positions)
        free_length = excluded_length(
            self.system.n_particles,
            self.system.length,
            self.system.rod_length,
        )
        # In one ordering sector, reduce rods to point-like coordinates:
        #     y_i = x_i - i*a,   L' = L - N*a.
        # The sine-product form is the periodic hard-rod / free-fermion-like
        # benchmark structure discussed in [Mazzanti2008HardRods].
        y = x - self.system.rod_length * np.arange(self.system.n_particles)
        total = 0.0
        for i in range(self.system.n_particles - 1):
            dy = y[i + 1 :] - y[i]
            s = np.sin(np.pi * dy / free_length)
            if np.any(s <= 0):
                return float("-inf")
            total += float(np.sum(np.log(s)))
        return self.power * total
