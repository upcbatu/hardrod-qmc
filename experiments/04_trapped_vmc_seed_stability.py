from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hrdmc.analysis import summarize_replicate_metrics
from hrdmc.io.artifacts import ensure_dir, write_json
from trapped_vmc_common import TrappedVMCCase, run_trapped_vmc_case, trapped_case_slug


STABILITY_METRICS = (
    "acceptance_rate",
    "valid_snapshot_fraction",
    "sampled_density_integral_error",
    "lda_integrated_particles_error",
    "sampled_potential_energy_mean",
    "density_l2_error_vmc_vs_lda",
    "relative_density_l2_error_vmc_vs_lda",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trapped VMC seed-stability diagnostics.")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--omega", type=float, default=0.10)
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--burn-in", type=int, default=1_500)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--seeds", default="4101,4102,4103")
    parser.add_argument("--grid-extent", type=float, default=40.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    seeds = parse_seeds(args.seeds)
    case = TrappedVMCCase(n_particles=args.n_particles, omega=args.omega)
    case_id = trapped_case_slug(case)

    replicates = []
    arrays_by_seed = {}
    for seed in seeds:
        replicate_summary, replicate_arrays = run_trapped_vmc_case(
            case,
            rod_length=args.rod_length,
            n_steps=args.steps,
            burn_in=args.burn_in,
            thinning=args.thinning,
            step_size=args.step_size,
            seed=seed,
            grid_extent=args.grid_extent,
            n_bins=args.n_bins,
        )
        replicate_summary["replicate_seed"] = seed
        replicate_summary["artifacts"] = [f"{case_id}_seed{seed}_density_profiles.npz"]
        replicates.append(replicate_summary)
        arrays_by_seed[seed] = replicate_arrays

    stability = summarize_replicate_metrics(replicates, STABILITY_METRICS)
    summary = {
        "status": "completed",
        "benchmark_tier": "VMC diagnostic seed stability",
        "case_id": case_id,
        "replicate_count": len(replicates),
        "replicates": replicates,
        "stability_metrics": stability,
        "controls": {
            "n_particles": args.n_particles,
            "omega": args.omega,
            "rod_length": args.rod_length,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "thinning": args.thinning,
            "step_size": args.step_size,
            "seeds": seeds,
            "grid_extent": args.grid_extent,
            "n_bins": args.n_bins,
        },
        "interpretation": (
            "Seed-stability diagnostic only; large metric spread means VMC settings or "
            "trial parameters must be improved before using trapped outputs as evidence."
        ),
    }

    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or repo_root / "results" / "trapped_vmc_seed_stability")
        write_json(out_dir / "summary.json", summary)
        for seed, arrays in arrays_by_seed.items():
            np.savez(out_dir / f"{case_id}_seed{seed}_density_profiles.npz", **arrays)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
