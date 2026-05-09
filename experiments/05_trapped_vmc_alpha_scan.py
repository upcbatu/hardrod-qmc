from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hrdmc.analysis import summarize_replicate_metrics
from hrdmc.io.artifacts import ensure_dir, write_json
from trapped_vmc_common import TrappedVMCCase, run_trapped_vmc_case, trapped_case_slug

ALPHA_SCAN_METRICS = (
    "acceptance_rate",
    "valid_snapshot_fraction",
    "sampled_total_energy_mean",
    "sampled_kinetic_energy_mean",
    "sampled_trap_energy_mean",
    "sampled_potential_energy_mean",
    "density_l2_error_vmc_vs_lda",
    "relative_density_l2_error_vmc_vs_lda",
    "sampled_rms_radius",
    "rms_radius_error_vmc_vs_lda",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trapped VMC diagnostic alpha scan.")
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--omega", type=float, default=0.10)
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--alpha-multipliers", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--seeds", default="5101,5102")
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--thinning", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--grid-extent", type=float, default=40.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one float value is required")
    if any(value <= 0.0 for value in values):
        raise ValueError("alpha multipliers must be positive")
    return values


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def alpha_label(alpha_multiplier: float) -> str:
    return f"{alpha_multiplier:.2f}".replace(".", "p")


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    alpha_multipliers = parse_csv_floats(args.alpha_multipliers)
    seeds = parse_csv_ints(args.seeds)
    case = TrappedVMCCase(n_particles=args.n_particles, omega=args.omega)
    case_id = trapped_case_slug(case)

    scan = []
    replicates = []
    arrays_by_id = {}
    for alpha_multiplier in alpha_multipliers:
        alpha_replicates = []
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
                alpha_multiplier=alpha_multiplier,
            )
            replicate_id = f"{case_id}_alpha{alpha_label(alpha_multiplier)}_seed{seed}"
            replicate_summary["alpha_multiplier"] = alpha_multiplier
            replicate_summary["replicate_seed"] = seed
            replicate_summary["artifacts"] = [f"{replicate_id}_density_profiles.npz"]
            alpha_replicates.append(replicate_summary)
            replicates.append(replicate_summary)
            arrays_by_id[replicate_id] = replicate_arrays

        metric_summary = summarize_replicate_metrics(alpha_replicates, ALPHA_SCAN_METRICS)
        scan.append(
            {
                "alpha_multiplier": alpha_multiplier,
                "gaussian_alpha": alpha_replicates[0]["trial"]["gaussian_alpha"],
                "replicate_count": len(alpha_replicates),
                "metric_summary": metric_summary,
            }
        )

    summary = {
        "status": "completed",
        "benchmark_tier": "VMC diagnostic alpha scan",
        "case_id": case_id,
        "scan": scan,
        "replicates": replicates,
        "controls": {
            "n_particles": args.n_particles,
            "omega": args.omega,
            "rod_length": args.rod_length,
            "alpha_multipliers": alpha_multipliers,
            "seeds": seeds,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "thinning": args.thinning,
            "step_size": args.step_size,
            "grid_extent": args.grid_extent,
            "n_bins": args.n_bins,
        },
        "interpretation": (
            "Diagnostic alpha scan only. sampled_total_energy_mean is the VMC local-energy "
            "diagnostic for the current trapped trial, not a hostile-audited production "
            "benchmark. sampled_potential_energy_mean is kept as a backward-compatible "
            "alias for harmonic trap energy only."
        ),
    }

    if not args.no_write:
        out_dir = ensure_dir(args.output_dir or repo_root / "results" / "trapped_vmc_alpha_scan")
        write_json(out_dir / "summary.json", summary)
        for replicate_id, arrays in arrays_by_id.items():
            np.savez(out_dir / f"{replicate_id}_density_profiles.npz", **arrays)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
