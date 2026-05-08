from __future__ import annotations

import json
from pathlib import Path

from hrdmc.config import load_experiment_config
from hrdmc.estimators import (
    estimate_pair_distribution,
    estimate_static_structure_factor,
)
from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.monte_carlo.vmc import MetropolisVMC
from hrdmc.plotting import plot_pair_distribution, plot_structure_factor
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import (
    excluded_length,
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
)
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "experiments" / "configs" / "smoke.json"
    cfg = load_experiment_config(cfg_path)
    sys_cfg = cfg.system
    vmc_cfg = cfg.vmc
    out_dir = ensure_dir(repo_root / cfg.output_dir)

    system = HardRodSystem(
        n_particles=sys_cfg.n_particles,
        length=sys_cfg.length,
        rod_length=sys_cfg.rod_length,
    )
    trial = HardRodJastrowTrial(system=system, power=1.0, nearest_neighbor_only=True)
    sampler = MetropolisVMC(
        system=system,
        trial=trial,
        step_size=vmc_cfg.step_size,
        seed=vmc_cfg.seed,
    )
    result = sampler.run(
        n_steps=vmc_cfg.n_steps,
        burn_in=vmc_cfg.burn_in,
        thinning=vmc_cfg.thinning,
    )

    g = estimate_pair_distribution(result.snapshots, system, n_bins=80)
    s = estimate_static_structure_factor(result.snapshots, system, n_modes=24)
    e_finite = hard_rod_finite_ring_energy_per_particle(
        system.n_particles,
        system.length,
        system.rod_length,
    )
    e_thermo = hard_rod_energy_per_particle(system.density, system.rod_length)
    reduced_length = excluded_length(system.n_particles, system.length, system.rod_length)

    artifacts = []
    plotting_status = "ok"
    try:
        plot_pair_distribution(g, out_dir / "g_of_r.png")
        plot_structure_factor(s, out_dir / "s_of_k.png")
        artifacts.extend(["g_of_r.png", "s_of_k.png"])
    except RuntimeError as exc:
        plotting_status = str(exc)

    # Numeric artifacts are always saved, even if matplotlib is unavailable.
    import numpy as np
    np.savez(out_dir / "observables.npz", r=g.r, g_r=g.g_r, k=s.k, s_k=s.s_k, s_k_stderr=s.stderr)
    artifacts.append("observables.npz")

    summary = {
        "status": "smoke prototype; not final DMC production result",
        "system": {
            "n_particles": system.n_particles,
            "length": system.length,
            "density": system.density,
            "rod_length": system.rod_length,
            "packing_fraction": system.packing_fraction,
            "excluded_length": reduced_length,
        },
        "vmc": result.metadata,
        "acceptance_rate": result.acceptance_rate,
        "n_snapshots": int(result.snapshots.shape[0]),
        "cpu_seconds": result.cpu_seconds,
        "reference_energy_per_particle_finite_N": e_finite,
        "reference_energy_per_particle_thermodynamic": e_thermo,
        "structure_factor_first_mode": float(s.s_k[0]),
        "structure_factor_first_mode_stderr": float(s.stderr[0]),
        "structure_factor_uncertainty_note": (
            "Per-mode stderr is computed across stored snapshots; blocking analysis is not "
            "reported in this smoke output."
        ),
        "plotting_status": plotting_status,
        "artifacts": artifacts,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
