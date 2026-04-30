from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hrdmc.analysis import density_l2_error
from hrdmc.estimators import estimate_open_line_density_profile
from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.monte_carlo.vmc import MetropolisVMC
from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.theory import lda_density_profile, lda_total_energy
from hrdmc.wavefunctions import TrappedHardRodTrial


@dataclass(frozen=True)
class TrappedGridCase:
    n_particles: int
    omega: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trapped hard-rod VMC diagnostic grid.")
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--grid-extent", type=float, default=40.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def default_cases() -> list[TrappedGridCase]:
    return [
        TrappedGridCase(n_particles=n_particles, omega=omega)
        for n_particles in (4, 8)
        for omega in (0.05, 0.10, 0.20)
    ]


def case_slug(case: TrappedGridCase) -> str:
    omega_label = f"{case.omega:.2f}".replace(".", "p")
    return f"N{case.n_particles}_omega{omega_label}"


def run_case(
    case: TrappedGridCase,
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
    sampled_integral = float(np.trapezoid(density.n_x, density.x))
    l2_error = density_l2_error(density.x, density.n_x, lda.n_x)
    valid_fraction = float(np.mean([system.is_valid(snapshot) for snapshot in result.snapshots]))
    slug = case_slug(case)

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


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    cases = []
    arrays_by_case = {}
    for index, case in enumerate(default_cases()):
        case_summary, case_arrays = run_case(
            case,
            rod_length=args.rod_length,
            n_steps=args.steps,
            burn_in=args.burn_in,
            thinning=args.thinning,
            step_size=args.step_size,
            seed=args.seed + index,
            grid_extent=args.grid_extent,
            n_bins=args.n_bins,
        )
        cases.append(case_summary)
        arrays_by_case[str(case_summary["case_id"])] = case_arrays

    summary = {
        "status": "completed",
        "benchmark_tier": "VMC diagnostic grid",
        "case_count": len(cases),
        "cases": cases,
        "controls": {
            "rod_length": args.rod_length,
            "n_particles": [4, 8],
            "omega": [0.05, 0.10, 0.20],
            "steps": args.steps,
            "burn_in": args.burn_in,
            "thinning": args.thinning,
            "step_size": args.step_size,
            "seed": args.seed,
            "grid_extent": args.grid_extent,
            "n_bins": args.n_bins,
        },
        "interpretation": (
            "Diagnostic VMC grid only; use it to check trapped density and LDA-grid "
            "plumbing before DMC/reference validation."
        ),
    }

    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or repo_root / "results" / "trapped_vmc_grid")
        write_json(out_dir / "summary.json", summary)
        for slug, arrays in arrays_by_case.items():
            np.savez(out_dir / f"{slug}_density_profiles.npz", **arrays)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
