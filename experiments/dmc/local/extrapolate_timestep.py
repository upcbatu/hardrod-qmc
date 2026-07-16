from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.timestep_extrapolation import (
    EnergyReportingResolutionPolicy,
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
        "--energy-reporting-resolution",
        type=float,
        help=(
            "Explicit practical energy resolution used only to bound an otherwise "
            "adequate and stable leading-model difference."
        ),
    )
    parser.add_argument(
        "--energy-reporting-confidence",
        type=float,
        help="Confidence level for model-order and fit-window upper allowances.",
    )
    parser.add_argument(
        "--energy-reporting-unit",
        help="Energy unit of the practical resolution; must match every input summary.",
    )
    parser.add_argument(
        "--energy-reporting-rationale",
        help="Scientific reporting rationale for the practical resolution.",
    )
    parser.add_argument(
        "--energy-reporting-policy-timing",
        choices=("prospective", "retrospective"),
        help="Whether the numerical resolution was fixed before or after inspecting these data.",
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
    energy_reporting_policy = _energy_reporting_policy(args, parser)
    payload = run_timestep_extrapolation_workflow(
        args.summary_paths,
        output_dir=args.output_dir,
        command=sys.argv,
        write_artifacts=not args.no_write,
        sensitivity_sigma=args.sensitivity_sigma,
        fit_alpha=args.fit_alpha,
        energy_assessment_manifest=args.energy_assessment_manifest,
        energy_reporting_policy=energy_reporting_policy,
    )
    extrapolation = payload["extrapolation"]
    artifacts = payload["workflow_artifacts"]
    if payload["publication_ready_within_fixed_population_timestep_scope"]:
        energy_summary = {
            "extrapolated_energy": payload["extrapolated_energy"],
            "statistical_stderr": payload["extrapolated_energy_statistical_stderr"],
            "leading_model_intercept_spread": extrapolation["leading_model_intercept_spread"],
        }
        if "extrapolated_energy_model_order_upper_allowance" in payload:
            energy_summary.update(
                {
                    "model_order_upper_allowance": payload[
                        "extrapolated_energy_model_order_upper_allowance"
                    ],
                    "fit_window_upper_allowance": payload[
                        "extrapolated_energy_fit_window_upper_allowance"
                    ],
                }
            )
    else:
        energy_summary = {
            "candidate_zero_step_energy": extrapolation["candidate_zero_step_energy"],
            "candidate_statistical_stderr": extrapolation[
                "candidate_zero_step_energy_statistical_stderr"
            ],
            "leading_model_intercept_spread": extrapolation["leading_model_intercept_spread"],
        }
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


def _energy_reporting_policy(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> EnergyReportingResolutionPolicy | None:
    values = {
        "--energy-reporting-resolution": args.energy_reporting_resolution,
        "--energy-reporting-confidence": args.energy_reporting_confidence,
        "--energy-reporting-unit": args.energy_reporting_unit,
        "--energy-reporting-rationale": args.energy_reporting_rationale,
        "--energy-reporting-policy-timing": args.energy_reporting_policy_timing,
    }
    supplied = [name for name, value in values.items() if value is not None]
    if not supplied:
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        parser.error(
            "energy reporting policy flags must be supplied together; missing " + ", ".join(missing)
        )
    try:
        return EnergyReportingResolutionPolicy(
            resolution=float(args.energy_reporting_resolution),
            confidence_level=float(args.energy_reporting_confidence),
            energy_unit=str(args.energy_reporting_unit),
            rationale=str(args.energy_reporting_rationale),
            timing=str(args.energy_reporting_policy_timing),
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
