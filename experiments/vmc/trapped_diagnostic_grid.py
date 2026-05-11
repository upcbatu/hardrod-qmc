from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from common import TrappedVMCCase, run_trapped_vmc_case

from hrdmc.io.artifacts import ensure_dir, write_json


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


def default_cases() -> list[TrappedVMCCase]:
    return [
        TrappedVMCCase(n_particles=n_particles, omega=omega)
        for n_particles in (4, 8)
        for omega in (0.05, 0.10, 0.20)
    ]


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    cases = []
    arrays_by_case = {}
    for index, case in enumerate(default_cases()):
        case_summary, case_arrays = run_trapped_vmc_case(
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
