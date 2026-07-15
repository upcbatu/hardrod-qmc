from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.benchmark_packet.matrix_assembly import (
    REQUIRED_CASE_ORDER,
    assemble_final_benchmark_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the canonical eight-case DMC benchmark matrix from verified "
            "primary packets and optional observable-specific supplements."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--r2-supplement",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="Use a verified packet as the R2/RMS source for one case.",
    )
    parser.add_argument(
        "--retrospective-energy-cases",
        default="",
        help=(
            "Comma-separated source cases whose energy status may use the recorded "
            "retrospective matrix-level stationarity assessment."
        ),
    )
    parser.add_argument("--energy-confidence-level", type=float, default=0.95)
    parser.add_argument("--energy-rhat-limit", type=float, default=1.01)
    parser.add_argument("--energy-min-effective-samples", type=float, default=400.0)
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        repo_root_from(Path(__file__))
        supplements = _parse_case_paths(args.r2_supplement)
        retrospective_cases = _parse_optional_cases(args.retrospective_energy_cases)
        payload, artifacts = assemble_final_benchmark_matrix(
            args.source_root,
            args.output_root,
            cases=REQUIRED_CASE_ORDER,
            r2_supplements=supplements,
            retrospective_energy_cases=retrospective_cases,
            energy_confidence_level=args.energy_confidence_level,
            energy_rhat_limit=args.energy_rhat_limit,
            energy_min_effective_samples=args.energy_min_effective_samples,
            command=sys.argv,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"assemble_final_matrix: {exc}") from None
    print_run_summary(
        run="assemble_final_matrix",
        status=str(payload["status"]),
        summary={
            "case_count": len(payload["rows"]),
            "accepted_count": sum(row["status"] == "accepted" for row in payload["rows"]),
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


def _parse_case_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise ValueError("R2 supplements must use CASE=PATH")
        if case_id in result:
            raise ValueError(f"duplicate R2 supplement case: {case_id}")
        result[case_id] = Path(path)
    return result


def _parse_optional_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(cases) != len(set(cases)):
        raise ValueError("retrospective energy case ids must be unique")
    return cases


if __name__ == "__main__":
    main()
