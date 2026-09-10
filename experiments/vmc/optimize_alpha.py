from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from hrdmc.artifacts.terminal import print_run_summary
from hrdmc.production.alpha_optimization import (
    AlphaOptimizationControls,
    run_alpha_optimization,
    write_alpha_optimization_outputs,
)
from hrdmc.system.settings import parse_case


def build_parser(*, require_case: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the reduced-TG relative alpha by correlated-sampling local-energy variance."
        )
    )
    parser.add_argument("--case", required=require_case)
    parser.add_argument("--seed", type=int, default=9401)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--drift-limiter", choices=("none", "umrigar"), default="umrigar")
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--burn-in-steps", type=int, default=20_000)
    parser.add_argument("--production-steps", type=int, default=20_000)
    parser.add_argument("--sample-stride-steps", type=int, default=100)
    parser.add_argument("--grid-extent", type=float, default=90.0)
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument("--reference-relative-alpha", "--alpha-center", type=float, required=True)
    parser.add_argument("--alpha-log-half-width", type=float, default=0.2)
    parser.add_argument("--alpha-grid-points", type=int, default=31)
    parser.add_argument("--min-reweight-ess-fraction", type=float, default=0.10)
    parser.add_argument("--max-configurations", type=int, default=100_000)
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    execute(build_parser().parse_args())


def execute(args: argparse.Namespace) -> None:
    case = parse_case(args.case)
    if case.rod_length == 0.0:
        raise ValueError("the exact hard-point guide does not require alpha optimization")
    controls = AlphaOptimizationControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_in_steps=args.burn_in_steps,
        production_steps=args.production_steps,
        sample_stride_steps=args.sample_stride_steps,
        drift_limiter=args.drift_limiter,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        reference_relative_alpha=args.reference_relative_alpha,
        alpha_log_half_width=args.alpha_log_half_width,
        alpha_grid_points=args.alpha_grid_points,
        min_reweight_ess_fraction=args.min_reweight_ess_fraction,
        max_configurations=args.max_configurations,
    )
    controls.validate()
    if case.rod_length == 0.0:
        raise ValueError("the exact hard-point guide does not require alpha optimization")
    if args.dry_run:
        _print_plan(case.case_id, args.seed, controls, args.output_dir, args.verbose_json)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {args.output_dir}")
    summary, rows, sample = run_alpha_optimization(case, controls, args.seed)
    artifacts = write_alpha_optimization_outputs(args.output_dir, summary, rows, sample)
    optimum = summary["candidate_metrics"]
    print_run_summary(
        run="optimize_relative_alpha",
        status=str(summary["status"]),
        summary={
            "case": case.case_id,
            "reference_relative_alpha": controls.reference_relative_alpha,
            "recommended_relative_alpha": optimum["relative_alpha"],
            "reweight_ess_fraction": optimum["reweight_ess_fraction"],
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=summary,
        verbose_json=args.verbose_json,
    )


def _print_plan(
    case_id: str,
    seed: int,
    controls: AlphaOptimizationControls,
    output_dir: Path,
    verbose_json: bool,
) -> None:
    plan = {
        "case_id": case_id,
        "seed": seed,
        "controls": asdict(controls),
        "stored_configuration_count": controls.stored_configuration_count,
        "output_dir": str(output_dir),
    }
    print_run_summary(
        run="optimize_relative_alpha",
        status="planned",
        summary={
            "case": case_id,
            "stored_configuration_count": controls.stored_configuration_count,
            "reference_relative_alpha": controls.reference_relative_alpha,
        },
        artifacts={"output_dir": str(output_dir)},
        verbose_payload=plan,
        verbose_json=verbose_json,
    )


if __name__ == "__main__":
    main()
