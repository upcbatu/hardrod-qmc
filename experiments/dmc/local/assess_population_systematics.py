from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.population_systematics import (
    FIXED_ENERGY_REPORTING_RESOLUTION,
    run_population_systematics_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify manifest-bound DMC mixed-energy summaries and assess a W/2W "
            "or W/2,W,2W walker-population ladder. A coarse-time-step W/2W pair "
            "is required to qualify the timestep-population interaction for "
            "publication readiness. When two timesteps are supplied, --selected-dt "
            "must identify the treatment being qualified."
        )
    )
    parser.add_argument(
        "summary_paths",
        nargs="+",
        type=Path,
        metavar="SUMMARY",
        help=(
            "DMC benchmark-packet or one-case stationarity summary. Supply two or "
            "three populations at the reference timestep. Add W/2W at one larger "
            "timestep to assess the required four-corner interaction; both timestep "
            "ladders are analyzed."
        ),
    )
    parser.add_argument(
        "--energy-reporting-resolution",
        type=float,
        required=True,
        help=(
            "Predeclared absolute energy resolution. The current thesis policy is "
            f"fixed at {FIXED_ENERGY_REPORTING_RESOLUTION:g} hbar*Omega."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; required unless --no-write is used.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for population-difference upper allowances.",
    )
    parser.add_argument(
        "--fit-alpha",
        type=float,
        default=0.05,
        help="Goodness-of-fit rejection level for the optional inverse-population fit.",
    )
    parser.add_argument(
        "--interaction-dt",
        type=float,
        help="Larger timestep represented by the second W/2W interaction pair.",
    )
    parser.add_argument(
        "--selected-dt",
        type=float,
        help=(
            "Timestep treatment whose population bound or population-limit energy "
            "is qualified. Required when two timesteps are supplied."
        ),
    )
    parser.add_argument(
        "--energy-assessment-manifest",
        type=Path,
        help=(
            "Optional verified final-matrix manifest that retrospectively qualifies "
            "exactly one supplied anchor energy by path and digest."
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
    payload = run_population_systematics_workflow(
        args.summary_paths,
        reporting_resolution=args.energy_reporting_resolution,
        output_dir=args.output_dir,
        command=sys.argv,
        write_artifacts=not args.no_write,
        confidence_level=args.confidence_level,
        fit_alpha=args.fit_alpha,
        interaction_dt=args.interaction_dt,
        selected_dt=args.selected_dt,
        energy_assessment_manifest=args.energy_assessment_manifest,
    )
    artifacts = payload["workflow_artifacts"]
    summary = {
        "case": payload["case_id"],
        "classification": payload["classification"],
        "selected_dt": payload["selected_dt"],
        "selected_dt_basis": payload["selected_dt_basis"],
        "selected_walkers": payload["selected_walkers"],
        "selected_last_doubling_upper_allowance": payload["selected_population_ladder"][
            "last_doubling"
        ]["upper_allowance"],
    }
    if "population_limit_energy_at_selected_timestep" in payload:
        summary["population_limit_energy_at_selected_timestep"] = payload[
            "population_limit_energy_at_selected_timestep"
        ]
        summary["population_limit_correction_at_selected_timestep"] = payload[
            "population_limit_correction_at_selected_timestep"
        ]
    if "finite_population_energy_at_selected_timestep" in payload:
        summary["finite_population_energy_at_selected_timestep"] = payload[
            "finite_population_energy_at_selected_timestep"
        ]
    print_run_summary(
        run="dmc_population_systematics",
        status=str(payload["status"]),
        summary=summary,
        artifacts={
            "summary": artifacts.get("summary"),
            "point_table": artifacts.get("point_table"),
            "comparison_table": artifacts.get("comparison_table"),
            "run_manifest": artifacts.get("run_manifest"),
            "output_dir": artifacts.get("output_dir"),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
