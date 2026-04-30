from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hrdmc.analysis import density_l2_error
from hrdmc.estimators import estimate_open_line_density_profile
from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.monte_carlo.vmc import MetropolisVMC
from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.theory import lda_density_profile, lda_total_energy
from hrdmc.wavefunctions import TrappedHardRodTrial


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trapped hard-rod VMC smoke test.")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--grid-extent", type=float, default=10.0)
    parser.add_argument("--n-bins", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    system = OpenLineHardRodSystem(
        n_particles=args.n_particles,
        rod_length=args.rod_length,
    )
    trap = HarmonicTrap(omega=args.omega)
    trial = TrappedHardRodTrial(
        system=system,
        gaussian_alpha=args.omega / np.sqrt(2.0),
        contact_power=1.0,
    )
    sampler = MetropolisVMC(
        system=system,
        trial=trial,
        step_size=args.step_size,
        seed=args.seed,
    )
    result = sampler.run(
        n_steps=args.steps,
        burn_in=args.burn_in,
        thinning=args.thinning,
    )

    x_min = -args.grid_extent
    x_max = args.grid_extent
    density = estimate_open_line_density_profile(
        result.snapshots,
        x_min=x_min,
        x_max=x_max,
        n_bins=args.n_bins,
    )
    potential_x = trap.values(density.x)
    lda = lda_density_profile(
        density.x,
        potential_x,
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    l2_error = density_l2_error(density.x, density.n_x, lda.n_x)
    sampled_density_integral = float(np.trapezoid(density.n_x, density.x))
    potential_values = np.asarray([trap.total(snapshot) for snapshot in result.snapshots], dtype=float)

    summary = {
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
        "valid_snapshot_fraction": float(np.mean([system.is_valid(x) for x in result.snapshots])),
        "sampled_density_integral": sampled_density_integral,
        "lda_integrated_particles": lda.integrated_particles,
        "lda_chemical_potential": lda.chemical_potential,
        "lda_total_energy": lda_total_energy(lda, rod_length=system.rod_length),
        "sampled_potential_energy_mean": float(np.mean(potential_values)),
        "sampled_potential_energy_stderr": float(
            np.std(potential_values, ddof=1) / np.sqrt(potential_values.size)
        ),
        "density_l2_error_vmc_vs_lda": l2_error,
        "grid": {
            "x_min": x_min,
            "x_max": x_max,
            "n_bins": args.n_bins,
        },
        "artifacts": ["summary.json", "density_profiles.npz"],
    }

    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or repo_root / "results" / "trapped_vmc_smoke")
        write_json(out_dir / "summary.json", summary)
        np.savez(
            out_dir / "density_profiles.npz",
            x=density.x,
            n_vmc=density.n_x,
            n_lda=lda.n_x,
            potential_x=potential_x,
            bin_edges=density.bin_edges,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
