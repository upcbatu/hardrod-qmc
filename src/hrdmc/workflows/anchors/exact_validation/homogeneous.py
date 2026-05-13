from __future__ import annotations

import numpy as np

from hrdmc.systems import excluded_length
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import (
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
)
from hrdmc.wavefunctions.trials import HardRodJastrowTrial
from hrdmc.workflows.anchors.exact_validation.models import HomogeneousRingAnchor


def run_homogeneous_ring_anchor(
    anchor: HomogeneousRingAnchor,
    *,
    rod_length: float,
    samples_per_case: int,
    seed: int,
    tolerance: float,
) -> dict[str, object]:
    if samples_per_case <= 0:
        raise ValueError("samples_per_case must be positive")
    density = anchor.packing_fraction / rod_length
    length = anchor.n_particles / density
    system = HardRodSystem(
        n_particles=anchor.n_particles,
        length=length,
        rod_length=rod_length,
    )
    trial = HardRodJastrowTrial(system=system, power=1.0, nearest_neighbor_only=False)
    exact_per_particle = hard_rod_finite_ring_energy_per_particle(
        n_particles=system.n_particles,
        length=system.length,
        rod_length=system.rod_length,
    )
    energies = np.asarray(
        [
            trial.local_kinetic_energy(positions)
            for positions in _sample_valid_ring_configurations(
                system,
                samples_per_case=samples_per_case,
                seed=seed,
            )
        ],
        dtype=float,
    )
    per_particle = energies / system.n_particles
    max_abs_error = float(np.max(np.abs(per_particle - exact_per_particle)))
    return {
        "anchor_id": anchor.anchor_id,
        "status": "passed" if max_abs_error <= tolerance else "failed",
        "anchor_type": "homogeneous_finite_a_ring_local_energy",
        "formula": "E_N/N = pi^2 * (N^2 - 1) / (3 * (L - N*a)^2)",
        "n_particles": system.n_particles,
        "rod_length": system.rod_length,
        "length": system.length,
        "packing_fraction": system.packing_fraction,
        "excluded_length": excluded_length(
            system.n_particles,
            system.length,
            system.rod_length,
        ),
        "samples": samples_per_case,
        "energy_total_mean": float(np.mean(energies)),
        "energy_total_exact_finite_N": exact_per_particle * system.n_particles,
        "energy_per_particle_mean": float(np.mean(per_particle)),
        "energy_per_particle_exact_finite_N": exact_per_particle,
        "energy_per_particle_thermodynamic": hard_rod_energy_per_particle(
            system.density,
            system.rod_length,
        ),
        "max_energy_per_particle_abs_error": max_abs_error,
        "tolerance_energy_per_particle_abs": tolerance,
    }


def _sample_valid_ring_configurations(
    system: HardRodSystem,
    *,
    samples_per_case: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    free_gap = system.length / system.n_particles - system.rod_length
    jitter = 0.08 * free_gap
    samples: list[np.ndarray] = []
    for _ in range(samples_per_case):
        positions = system.initial_lattice(
            jitter=jitter,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        positions = system.wrap(positions + rng.uniform(0.0, system.length))
        if not system.is_valid(positions):
            raise RuntimeError("generated invalid homogeneous ring configuration")
        samples.append(positions)
    return samples
