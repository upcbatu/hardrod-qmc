from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.fw_sensitivity import run_fw_sensitivity_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a manifest-bound transported forward-walking benchmark packet "
            "with the observable-composed fine-timestep final-matrix anchor."
        )
    )
    parser.add_argument(
        "--final-matrix-manifest",
        type=Path,
        required=True,
        help="Accepted final-matrix assembly run_manifest.json.",
    )
    parser.add_argument("--case", required=True, help="Active N*_A* case identifier.")
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        required=True,
        help="Candidate dmc_benchmark_packet_v3 summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory; required unless --no-write is used.",
    )
    parser.add_argument(
        "--rms-relative-margin",
        type=float,
        default=0.001,
        help="Paired RMS-radius relative equivalence margin.",
    )
    parser.add_argument(
        "--density-relative-l2-margin",
        type=float,
        default=0.03,
        help="Exact-grid paired density relative-L2 equivalence margin.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Family-wise confidence level for paired equivalence bounds.",
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
    payload = run_fw_sensitivity_workflow(
        args.final_matrix_manifest,
        args.candidate_summary,
        case_id=args.case,
        output_dir=args.output_dir,
        command=sys.argv,
        write_artifacts=not args.no_write,
        rms_relative_margin=args.rms_relative_margin,
        density_relative_l2_margin=args.density_relative_l2_margin,
        confidence_level=args.confidence_level,
    )
    comparison = payload.get("observable_comparison")
    summary: dict[str, object] = {
        "case": payload["case_id"],
        "publication_ready": payload["publication_ready_within_fw_sensitivity_scope"],
    }
    if isinstance(comparison, dict):
        density = comparison.get("density")
        rms = comparison.get("rms_radius")
        if isinstance(density, dict):
            summary["density_relative_l2_upper_bound"] = density.get("simultaneous_upper_bound")
        if isinstance(rms, dict):
            summary["rms_relative_upper_bound"] = rms.get("simultaneous_relative_upper_bound")
    artifacts = payload["workflow_artifacts"]
    print_run_summary(
        run="dmc_fw_sensitivity",
        status=str(payload["status"]),
        summary=summary,
        artifacts={
            "summary": artifacts.get("summary"),
            "observable_table": artifacts.get("observable_table"),
            "shell_table": artifacts.get("shell_table"),
            "run_manifest": artifacts.get("run_manifest"),
            "output_dir": artifacts.get("output_dir"),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


if __name__ == "__main__":
    main()
