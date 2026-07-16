from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.energy_stationarity_assessment import (
    run_energy_stationarity_assessment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one simultaneous mixed-energy stationarity assessment to a "
            "predeclared family of manifest-bound DMC benchmark packets."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="CASE=SUMMARY",
        help="Case id and benchmark summary path; repeat once per declared case.",
    )
    parser.add_argument(
        "--expected-cases",
        required=True,
        help="Comma-separated case order that the supplied source family must match exactly.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--rhat-limit", type=float, default=1.05)
    parser.add_argument("--min-effective-samples", type=float, default=30.0)
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root_from(Path(__file__))
    expected_cases = tuple(item for item in args.expected_cases.split(",") if item)
    sources = _parse_sources(args.source)
    payload = run_energy_stationarity_assessment(
        sources,
        expected_case_ids=expected_cases,
        output_dir=args.output_dir,
        command=sys.argv,
        confidence_level=args.confidence_level,
        rhat_limit=args.rhat_limit,
        min_effective_samples=args.min_effective_samples,
    )
    artifacts = payload["workflow_artifacts"]
    print_run_summary(
        run="dmc_energy_stationarity_assessment",
        status=str(payload["status"]),
        summary={
            "case_count": len(expected_cases),
            "publication_ready": payload[
                "publication_ready_within_energy_stationarity_scope"
            ],
        },
        artifacts={
            "summary": artifacts["summary"],
            "case_table": artifacts["case_table"],
            "run_manifest": artifacts["run_manifest"],
            "output_dir": artifacts["output_dir"],
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


def _parse_sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise ValueError("--source values must use CASE=SUMMARY")
        if case_id in result:
            raise ValueError(f"duplicate --source case: {case_id}")
        result[case_id] = Path(path)
    return result


if __name__ == "__main__":
    main()
