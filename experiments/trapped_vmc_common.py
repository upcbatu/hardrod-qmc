from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrdmc.analysis import density_l2_error, relative_density_l2_error
from hrdmc.estimators import estimate_open_line_density_profile, integrate_density_profile
from hrdmc.monte_carlo.vmc import MetropolisVMC
from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.theory import lda_density_profile, lda_total_energy
from hrdmc.wavefunctions import TrappedHardRodTrial


@dataclass(frozen=True)
class TrappedVMCCase:
    n_particles: int
    omega: float


def trapped_case_slug(case: TrappedVMCCase) -> str:
    omega_label = f"{case.omega:.2f}".replace(".", "p")
    return f"N{case.n_particles}_omega{omega_label}"


def run_trapped_vmc_case(
    case: TrappedVMCCase,
    *,
    rod_length: float,
    n_steps: int,
    burn_in: int,
    thinning: int,
    step_size: float,
    seed: int,
    grid_extent: float,
    n_bins: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    system = OpenLineHardRodSystem(n_particles=case.n_particles, rod_length=rod_length)
    trap = HarmonicTrap(omega=case.omega)
    trial = TrappedHardRodTrial(
        system=system,
        gaussian_alpha=case.omega / np.sqrt(2.0),
        contact_power=1.0,
    )
    sampler = MetropolisVMC(system=system, trial=trial, step_size=step_size, seed=seed)
    result = sampler.run(n_steps=n_steps, burn_in=burn_in, thinning=thinning)

    density = estimate_open_line_density_profile(
        result.snapshots,
        x_min=-grid_extent,
        x_max=grid_extent,
        n_bins=n_bins,
    )
    potential_x = trap.values(density.x)
    lda = lda_density_profile(
        density.x,
        potential_x,
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    potential_values = np.asarray([trap.total(snapshot) for snapshot in result.snapshots], dtype=float)
    sampled_integral = integrate_density_profile(density)
    l2_error = density_l2_error(density.x, density.n_x, lda.n_x)
    relative_l2_error = relative_density_l2_error(density.x, density.n_x, lda.n_x)
    valid_fraction = float(np.mean([system.is_valid(snapshot) for snapshot in result.snapshots]))
    slug = trapped_case_slug(case)

    summary = {
        "case_id": slug,
        "status": "completed",
        "benchmark_tier": "VMC diagnostic",
        "system": {
            "geometry": "open_line",
            "n_particles": system.n_particles,
            "rod_length": system.rod_length,
        },
        "trap": {
            "type": "harmonic",
            "omega": trap.omega,
            "center": trap.center,
        },
        "trial": {
            "type": "TrappedHardRodTrial",
            "gaussian_alpha": trial.gaussian_alpha,
            "contact_power": trial.contact_power,
        },
        "vmc": result.metadata,
        "acceptance_rate": result.acceptance_rate,
        "n_snapshots": int(result.snapshots.shape[0]),
        "valid_snapshot_fraction": valid_fraction,
        "sampled_density_integral": sampled_integral,
        "sampled_density_integral_error": sampled_integral - system.n_particles,
        "lda_integrated_particles": lda.integrated_particles,
        "lda_integrated_particles_error": lda.integrated_particles - system.n_particles,
        "lda_chemical_potential": lda.chemical_potential,
        "lda_total_energy": lda_total_energy(lda, rod_length=system.rod_length),
        "sampled_potential_energy_mean": float(np.mean(potential_values)),
        "sampled_potential_energy_stderr": float(
            np.std(potential_values, ddof=1) / np.sqrt(potential_values.size)
        ),
        "density_l2_error_vmc_vs_lda": l2_error,
        "density_l2_error_units": "particles^2/length",
        "relative_density_l2_error_vmc_vs_lda": relative_l2_error,
        "grid": {
            "x_min": -grid_extent,
            "x_max": grid_extent,
            "n_bins": n_bins,
        },
        "artifacts": [f"{slug}_density_profiles.npz"],
    }
    arrays = {
        "x": density.x,
        "n_vmc": density.n_x,
        "n_lda": lda.n_x,
        "potential_x": potential_x,
        "bin_edges": density.bin_edges,
    }
    return summary, arrays
