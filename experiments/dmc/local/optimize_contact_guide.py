from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.io import print_run_summary, progress_bar, progress_requested
from hrdmc.workflows.dmc.guide_parameter_optimization import (
    ContactGuideOptimizationControls,
    run_contact_guide_optimization,
    write_contact_guide_optimization_outputs,
)
from hrdmc.workflows.dmc.trapped import parse_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize reduced-TG relative width and a contact-regular partial "
            "N=2 gap correction from one branching-free correlated sample."
        )
    )
    parser.add_argument("--case", default="N20_A10")
    parser.add_argument("--seed", type=int, default=9401)
    parser.add_argument("--dt", type=float, default=0.00025)
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--burn-tau", type=float, default=20.0)
    parser.add_argument("--sample-tau", type=float, default=10.0)
    parser.add_argument("--sample-stride-steps", type=int, default=200)
    parser.add_argument("--start-scale", type=float, default=1.0)
    parser.add_argument("--grid-extent", type=float, default=90.0)
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument("--reference-relative-alpha", type=float, default=7.01001132673)
    parser.add_argument("--reference-contact-beta", type=float, default=0.0)
    parser.add_argument("--alpha-log-half-width", type=float, default=0.2)
    parser.add_argument("--alpha-grid-points", type=int, default=31)
    parser.add_argument("--beta-initial-max", type=float, default=0.25)
    parser.add_argument("--beta-grid-points", type=int, default=26)
    parser.add_argument("--min-reweight-ess-fraction", type=float, default=0.10)
    parser.add_argument("--max-configurations", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root_from(Path(__file__))
    case = parse_case(args.case)
    controls = ContactGuideOptimizationControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        sample_tau=args.sample_tau,
        sample_stride_steps=args.sample_stride_steps,
        start_scale=args.start_scale,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        reference_relative_alpha=args.reference_relative_alpha,
        reference_contact_beta=args.reference_contact_beta,
        alpha_log_half_width=args.alpha_log_half_width,
        alpha_grid_points=args.alpha_grid_points,
        beta_initial_max=args.beta_initial_max,
        beta_grid_points=args.beta_grid_points,
        min_reweight_ess_fraction=args.min_reweight_ess_fraction,
        max_configurations=args.max_configurations,
    )
    controls.validate()
    if args.dry_run:
        plan = {
            "case_id": case.case_id,
            "seed": args.seed,
            "controls": to_jsonable(controls.__dict__),
            "total_steps": controls.total_steps,
            "stored_configuration_count": controls.stored_configuration_count,
            "output_dir": str(args.output_dir),
        }
        print_run_summary(
            run="optimize_contact_guide",
            status="planned",
            summary={
                "case": case.case_id,
                "total_steps": controls.total_steps,
                "stored_configuration_count": controls.stored_configuration_count,
                "reference_relative_alpha": controls.reference_relative_alpha,
                "reference_contact_beta": controls.reference_contact_beta,
            },
            artifacts={"output_dir": str(args.output_dir)},
            verbose_payload=plan,
            verbose_json=args.verbose_json,
        )
        return
    with progress_bar(
        total=controls.total_steps,
        label=f"Contact-guide optimization {case.case_id}",
        enabled=progress_requested(args.progress),
    ) as bar:
        summary, rows, sample = run_contact_guide_optimization(
            case,
            controls,
            args.seed,
            progress=bar,
        )
    artifacts = write_contact_guide_optimization_outputs(
        args.output_dir,
        summary=summary,
        rows=rows,
        sample=sample,
        command=sys.argv,
    )
    selected = summary.get("recommended_parameters", {})
    metrics = summary.get("candidate_metrics", {})
    print_run_summary(
        run="optimize_contact_guide",
        status=str(summary.get("status", "completed")),
        summary={
            "case": case.case_id,
            "relative_alpha": selected.get("relative_alpha"),
            "contact_beta": selected.get("contact_beta"),
            "reweight_ess_fraction": metrics.get("reweight_ess_fraction"),
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=summary,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
