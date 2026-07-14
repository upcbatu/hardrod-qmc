from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hrdmc.artifacts import repo_root_from
from hrdmc.io import print_run_summary
from hrdmc.workflows.dmc.benchmark_packet.reanalysis import (
    reanalyze_benchmark_packet,
    write_benchmark_reanalysis_matrix,
)
from hrdmc.workflows.dmc.final_matrix import DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE
from hrdmc.workflows.dmc.trapped import parse_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reclassify stored benchmark-packet forward-walking ladders with a "
            "source-bound simultaneous RMS-equivalence analysis."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids; by default discover every source summary.",
    )
    parser.add_argument(
        "--rms-relative-equivalence-margin",
        type=float,
        default=DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE,
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--sensitivity-margins", default="0.0001,0.0002,0.0005,0.001")
    parser.add_argument(
        "--policy-timing",
        choices=("prospective", "retrospective"),
        required=True,
        help="Record whether the equivalence rule preceded inspection of the source rows.",
    )
    parser.add_argument(
        "--prospective-cases",
        default="",
        help="Case ids whose policy timing overrides the default to prospective.",
    )
    parser.add_argument(
        "--retrospective-cases",
        default="",
        help="Case ids whose policy timing overrides the default to retrospective.",
    )
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root_from(Path(__file__))
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"reanalysis output root is not empty: {args.output_root}")
    cases = _resolve_cases(args.source_root, args.cases)
    policy_timing_by_case = _resolve_policy_timing(
        cases,
        default=args.policy_timing,
        prospective=_parse_optional_case_tuple(args.prospective_cases),
        retrospective=_parse_optional_case_tuple(args.retrospective_cases),
    )
    sensitivity_margins = _parse_float_tuple(args.sensitivity_margins)
    plot_formats = _parse_str_tuple(args.plot_formats)
    rows = []
    artifacts_by_case: dict[str, dict[str, Path]] = {}
    for case_id in cases:
        payload, artifacts = reanalyze_benchmark_packet(
            args.source_root / case_id,
            args.output_root / case_id,
            rms_relative_equivalence_margin=args.rms_relative_equivalence_margin,
            confidence_level=args.confidence_level,
            sensitivity_margins=sensitivity_margins,
            policy_timing=policy_timing_by_case[case_id],
            plot_formats=plot_formats,
            write_plots=not args.skip_plots,
            command=sys.argv,
        )
        rows.append(
            _matrix_row(
                case_id=case_id,
                payload=payload,
                policy_timing=policy_timing_by_case[case_id],
            )
        )
        artifacts_by_case[case_id] = artifacts
    matrix_artifacts = write_benchmark_reanalysis_matrix(
        args.output_root,
        source_root=args.source_root,
        cases=cases,
        rows=rows,
        artifacts_by_case=artifacts_by_case,
        rms_relative_equivalence_margin=args.rms_relative_equivalence_margin,
        confidence_level=args.confidence_level,
        sensitivity_margins=sensitivity_margins,
        policy_timing_by_case=policy_timing_by_case,
        plot_formats=plot_formats,
        write_plots=not args.skip_plots,
        command=sys.argv,
    )
    matrix_status = "accepted" if all(row["status"] == "accepted" for row in rows) else "review"
    print_run_summary(
        run="reanalyze_benchmark_packet",
        status=matrix_status,
        summary={
            "case_count": len(rows),
            "accepted_count": sum(row["status"] == "accepted" for row in rows),
        },
        artifacts={
            "output_root": str(args.output_root),
            **{name: str(path) for name, path in matrix_artifacts.items()},
        },
        verbose_payload={
            "rows": rows,
            "artifacts_by_case": {
                case_id: {name: str(path) for name, path in artifacts.items()}
                for case_id, artifacts in artifacts_by_case.items()
            },
        },
        verbose_json=args.verbose_json,
    )


def _matrix_row(
    *,
    case_id: str,
    payload: dict[str, object],
    policy_timing: str,
) -> dict[str, object]:
    estimates = _mapping(payload.get("estimates"))
    energy = _mapping(estimates.get("energy"))
    rms = _mapping(estimates.get("rms"))
    density = _mapping(estimates.get("density"))
    controls = _mapping(payload.get("controls"))
    return {
        "case": case_id,
        "status": payload["status"],
        "energy_status": payload["energy_validation_status"],
        "pure_fw_status": payload["pure_fw_validation_status"],
        "policy_timing": policy_timing,
        "guide_family": payload.get("guide_family"),
        "dt": controls.get("dt"),
        "walkers": controls.get("walkers"),
        "energy": energy.get("value"),
        "energy_stderr": energy.get("stderr"),
        "energy_relative_delta_vs_lda": _relative_delta(energy),
        "rms": rms.get("value"),
        "rms_mc_statistical_stderr": rms.get("mc_statistical_stderr"),
        "rms_fw_lag_systematic_relative_upper_bound": rms.get(
            "fw_lag_systematic_relative_upper_bound"
        ),
        "rms_relative_delta_vs_lda": _relative_delta(rms),
        "density_status": density.get("status"),
        "fw_density_relative_l2_vs_lda": density.get("fw_relative_l2_vs_lda"),
        "fw_density_relative_l2_vs_mixed": density.get("fw_relative_l2_vs_mixed"),
        "density_fw_lag_systematic_relative_l2_upper_bound": density.get(
            "fw_lag_systematic_relative_l2_upper_bound"
        ),
    }


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _relative_delta(estimate: dict[str, object]) -> float | None:
    delta = estimate.get("delta_vs_lda")
    reference = estimate.get("lda_value")
    if not isinstance(delta, (int, float)) or not isinstance(reference, (int, float)):
        return None
    if reference == 0.0:
        return None
    return float(delta / reference)


def _resolve_cases(source_root: Path, value: str | None) -> tuple[str, ...]:
    if value is not None:
        cases = _parse_str_tuple(value)
    else:
        cases = tuple(path.parent.name for path in sorted(source_root.glob("N*_A*/summary.json")))
    if not cases:
        raise ValueError("no benchmark-packet source cases found")
    if len(cases) != len(set(cases)):
        raise ValueError("case ids must be unique")
    for case_id in cases:
        parsed = parse_case(case_id)
        if parsed.case_id != case_id:
            raise ValueError(f"case id is not canonical: {case_id}")
    return cases


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one value is required")
    return values


def _parse_optional_case_tuple(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return _parse_str_tuple(value)


def _resolve_policy_timing(
    cases: tuple[str, ...],
    *,
    default: str,
    prospective: tuple[str, ...],
    retrospective: tuple[str, ...],
) -> dict[str, str]:
    if default not in {"prospective", "retrospective"}:
        raise ValueError("default policy timing is invalid")
    requested = set(cases)
    prospective_set = set(prospective)
    retrospective_set = set(retrospective)
    if len(prospective_set) != len(prospective) or len(retrospective_set) != len(retrospective):
        raise ValueError("policy timing override case ids must be unique")
    overlap = prospective_set & retrospective_set
    if overlap:
        raise ValueError("policy timing override sets overlap: " + ", ".join(sorted(overlap)))
    unknown = (prospective_set | retrospective_set) - requested
    if unknown:
        raise ValueError(
            "policy timing override cases were not requested: " + ", ".join(sorted(unknown))
        )
    for case_id in prospective_set | retrospective_set:
        if parse_case(case_id).case_id != case_id:
            raise ValueError(f"policy timing case id is not canonical: {case_id}")
    result = {case_id: default for case_id in cases}
    result.update({case_id: "prospective" for case_id in prospective_set})
    result.update({case_id: "retrospective" for case_id in retrospective_set})
    return result


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_str_tuple(value))


if __name__ == "__main__":
    main()
