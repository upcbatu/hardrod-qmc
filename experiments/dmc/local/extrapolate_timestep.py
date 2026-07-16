from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.timestep_extrapolation import (
    run_timestep_extrapolation_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify manifest-bound DMC energy summaries, compare leading-linear "
            "and leading-quadratic zero-step models, and report curvature diagnostics."
        )
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        type=Path,
        metavar="SUMMARY",
        help="DMC benchmark-packet or one-case stationarity summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; required unless --no-write is used.",
    )
    parser.add_argument(
        "--sensitivity-sigma",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to declared-error fit uncertainty when classifying "
            "model and largest-point sensitivity (default: 2)."
        ),
    )
    parser.add_argument(
        "--fit-alpha",
        type=float,
        default=0.05,
        help="Goodness-of-fit rejection level applied to declared point errors.",
    )
    parser.add_argument(
        "--energy-assessment-manifest",
        type=Path,
        help=(
            "Manifest-bound final-matrix assessment selecting one exact input "
            "summary for energy-quality classification."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Verify and analyze inputs without writing artifacts.",
    )
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root_from(Path(__file__))
    if not args.no_write and args.output_dir is None:
        parser.error("--output-dir is required unless --no-write is used")
    payload = run_timestep_extrapolation_workflow(
        args.summary_paths,
        output_dir=args.output_dir,
        command=sys.argv,
        write_artifacts=not args.no_write,
        sensitivity_sigma=args.sensitivity_sigma,
        fit_alpha=args.fit_alpha,
        energy_assessment_manifest=args.energy_assessment_manifest,
    )
    extrapolation = payload["extrapolation"]
    artifacts = payload["workflow_artifacts"]
    energy_summary = (
        {
            "extrapolated_energy": payload["extrapolated_energy"],
            "statistical_stderr": payload["extrapolated_energy_statistical_stderr"],
            "model_spread_allowance": extrapolation["model_spread_systematic_allowance"],
        }
        if payload["status"] in {"accepted", "accepted_with_warnings"}
        else {
            "candidate_zero_step_energy": extrapolation["candidate_zero_step_energy"],
            "candidate_statistical_stderr": extrapolation[
                "candidate_zero_step_energy_statistical_stderr"
            ],
            "model_spread_allowance": extrapolation["model_spread_systematic_allowance"],
        }
    )
    print_run_summary(
        run="dmc_timestep_extrapolation",
        status=str(payload["status"]),
        summary={
            "case": payload["case_id"],
            "point_count": payload["point_count"],
            **energy_summary,
        },
        artifacts={
            "summary": artifacts.get("summary"),
            "point_table": artifacts.get("point_table"),
            "run_manifest": artifacts.get("run_manifest"),
            "output_dir": artifacts.get("output_dir"),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
