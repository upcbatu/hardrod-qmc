from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts.layout import repo_root_from
from hrdmc.artifacts.terminal import print_run_summary
from hrdmc.uncertainty.budget import (
    SYSTEMATIC_LANES,
    assemble_numerical_systematics_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the canonical eight-case DMC result package from one accepted "
            "final-matrix manifest and manifest-bound timestep, walker-population, "
            "and forward-walking sensitivity assessments. Missing assessments remain "
            "visible and are excluded from the thesis-facing energy table."
        )
    )
    parser.add_argument(
        "--final-matrix-manifest",
        type=Path,
        required=True,
        help="Accepted final-matrix assembly run_manifest.json.",
    )
    parser.add_argument(
        "--timestep",
        action="append",
        default=[],
        metavar="CASE=MANIFEST",
        help="Manifest-bound zero-time-step assessment for one finite-A case.",
    )
    parser.add_argument(
        "--population",
        action="append",
        default=[],
        metavar="CASE=MANIFEST",
        help="Manifest-bound selected-treatment walker-population assessment.",
    )
    parser.add_argument(
        "--fw-sensitivity",
        action="append",
        default=[],
        metavar="CASE=MANIFEST",
        help="Manifest-bound transported forward-walking sensitivity assessment.",
    )
    parser.add_argument(
        "--bounded-qualifier",
        action="append",
        default=[],
        metavar="CASE:LANE=RATIONALE",
        help=(
            "Explicit thesis qualifier for a source assessment that is numerically "
            "bounded but not directly accepted. LANE is timestep, population, or "
            "forward_walking; the source artifact must independently establish the bound."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        repo_root_from(Path(__file__))
        payload, artifacts = assemble_numerical_systematics_package(
            args.final_matrix_manifest,
            timestep_manifests=_parse_case_manifests(args.timestep, name="timestep"),
            population_manifests=_parse_case_manifests(args.population, name="population"),
            fw_sensitivity_manifests=_parse_case_manifests(
                args.fw_sensitivity,
                name="forward-walking sensitivity",
            ),
            output_dir=args.output_dir,
            bounded_qualifiers=_parse_bounded_qualifiers(args.bounded_qualifier),
            command=sys.argv,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"assemble_numerical_systematics: {exc}") from None
    print_run_summary(
        run="dmc_numerical_systematics_package",
        status=str(payload["status"]),
        summary={
            "publication_ready_cases": payload["publication_ready_case_count"],
            "case_count": len(payload["rows"]),
            "unresolved_cases": payload["unresolved_cases"],
            "missing_inputs": payload["missing_inputs"],
        },
        artifacts={name: str(path) for name, path in artifacts.items()},
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )


def _parse_case_manifests(values: list[str], *, name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        case_id, separator, path = value.partition("=")
        if not separator or not case_id or not path:
            raise ValueError(f"{name} inputs must use CASE=MANIFEST")
        if case_id in result:
            raise ValueError(f"duplicate {name} case: {case_id}")
        result[case_id] = Path(path)
    return result


def _parse_bounded_qualifiers(values: list[str]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for value in values:
        selection, separator, rationale = value.partition("=")
        case_id, lane_separator, lane = selection.partition(":")
        if (
            not separator
            or not lane_separator
            or not case_id
            or lane not in SYSTEMATIC_LANES
            or not rationale.strip()
        ):
            raise ValueError(
                "bounded qualifiers must use CASE:LANE=RATIONALE with a supported lane"
            )
        key = (case_id, lane)
        if key in result:
            raise ValueError(f"duplicate bounded qualifier: {case_id}:{lane}")
        result[key] = rationale.strip()
    return result


if __name__ == "__main__":
    main()
