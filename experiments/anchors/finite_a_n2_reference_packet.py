from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io import print_run_summary, progress_requested
from hrdmc.plotting.figures.finite_a_n2_reference import (
    write_finite_a_n2_reference_plots,
)
from hrdmc.workflows.anchors.finite_a_n2 import (
    FiniteAN2ReferenceTolerances,
    summarize_finite_a_n2_reference_case,
    write_finite_a_n2_reference_artifacts,
    write_finite_a_n2_reference_manifest,
)
from hrdmc.workflows.dmc.collective_rn import (
    DEFAULT_COMPONENT_LOG_SCALES,
    DEFAULT_COMPONENT_PROBABILITIES,
    DEFAULT_PROPOSAL_FAMILY,
    DEFAULT_TARGET_FAMILY,
    PROPOSAL_FAMILIES,
    TARGET_FAMILIES,
    CollectiveRNControls,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    GUIDE_FAMILIES,
    DMCRunControls,
    TrappedCase,
    controls_to_dict,
    dmc_progress_bar,
    resolve_parallel_workers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N=2 finite-A trapped DMC/FW cases against a deterministic "
            "relative-coordinate reference."
        )
    )
    parser.add_argument(
        "--rod-length-values",
        default="0.2",
        help="Comma-separated A=a/a_ho values in harmonic-oscillator length units.",
    )
    parser.add_argument("--seeds", default="1001,1002")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=256)
    collective = parser.add_argument_group("optional collective RN move")
    collective.add_argument("--collective-rn", action="store_true")
    collective.add_argument("--collective-cadence-tau", type=float, default=0.01)
    parser.add_argument(
        "--local-step-method",
        choices=("euler", "metropolis"),
        default="metropolis",
    )
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=240.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=20.0)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=800,
        help=(
            "Density histogram bins. The default is intentionally high enough "
            "to resolve finite-N trapped density peaks in report figures."
        ),
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument(
        "--relative-alpha",
        type=float,
        default=None,
        help=(
            "Optional reduced-coordinate internal Gaussian width for the "
            "reduced-TG guide. The center-of-mass width remains harmonic."
        ),
    )
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    collective.add_argument(
        "--proposal-family",
        choices=PROPOSAL_FAMILIES,
        default=DEFAULT_PROPOSAL_FAMILY,
    )
    parser.add_argument(
        "--guide-family",
        choices=GUIDE_FAMILIES,
        default=DEFAULT_GUIDE_FAMILY,
    )
    collective.add_argument(
        "--target-family",
        choices=TARGET_FAMILIES,
        default=DEFAULT_TARGET_FAMILY,
    )
    collective.add_argument(
        "--component-log-scales",
        default=_format_float_tuple(DEFAULT_COMPONENT_LOG_SCALES),
    )
    collective.add_argument(
        "--component-probabilities",
        default=_format_float_tuple(DEFAULT_COMPONENT_PROBABILITIES),
    )
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-invalid-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument("--pure-fw-lags", default="0,10,20,30,40,50")
    parser.add_argument("--pure-fw-density-lags", default=None)
    parser.add_argument("--pure-fw-observables", default="r2,density")
    parser.add_argument(
        "--pure-fw-observable-source",
        choices=("raw_r2", "r2_rb"),
        default="raw_r2",
    )
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-min-source-ancestor-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-max-source-family-fraction", type=float, default=0.25)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=1)
    parser.add_argument(
        "--pure-fw-density-collection-stride-steps",
        type=int,
        default=None,
    )
    parser.add_argument("--pure-fw-plateau-window-lag-count", type=int, default=4)
    parser.add_argument(
        "--density-plateau-relative-l2-tolerance",
        type=float,
        default=0.03,
    )
    parser.add_argument("--reference-grid-points", type=int, default=1400)
    parser.add_argument("--reference-y-max", type=float, default=None)
    parser.add_argument("--energy-abs-tolerance", type=float, default=0.02)
    parser.add_argument("--pure-r2-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--pure-rms-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--pure-density-l2-tolerance", type=float, default=0.10)
    parser.add_argument("--density-accounting-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--verbose-json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    controls = DMCRunControls(
        dt=args.dt,
        walkers=args.walkers,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
        local_step_method=args.local_step_method,
        relative_alpha=args.relative_alpha,
    )
    seeds = _parse_ints(args.seeds)
    cases = [
        TrappedCase(n_particles=2, rod_length=rod_length)
        for rod_length in _parse_floats(args.rod_length_values)
    ]
    requested_workers = resolve_parallel_workers(len(seeds), args.parallel_workers)
    payload = _build_payload(
        args=args,
        controls=controls,
        seeds=seeds,
        cases=cases,
        requested_workers=requested_workers,
    )
    output_dir: Path | None = None
    if not args.no_write:
        write_output_dir = Path(
            args.output_dir
            or artifact_dir(
                repo_root,
                ArtifactRoute("dmc", "local", "finite_a_n2_reference_packet"),
            )
        )
        output_dir = write_output_dir
        plot_paths: list[str] = []
        if not args.skip_plots:
            plot_paths = write_finite_a_n2_reference_plots(
                write_output_dir,
                payload,
                formats=_parse_str_tuple(args.plot_formats),
            )
            payload["plots"] = plot_paths
        artifacts = write_finite_a_n2_reference_artifacts(write_output_dir, payload)
        artifacts.extend(write_output_dir / path for path in plot_paths)
        write_finite_a_n2_reference_manifest(
            write_output_dir,
            payload,
            artifacts,
            controls,
            seeds,
            sys.argv,
        )
    print_run_summary(
        run="finite_a_n2_reference_packet",
        status=str(payload["status"]),
        summary={
            "case_count": len(cases),
            "seed_count": len(seeds),
            "accepted_cases": sum(row["status"] == "accepted" for row in payload["case_table"]),
        },
        artifacts={
            "summary": None if output_dir is None else str(output_dir / "summary.json"),
            "output_dir": None if output_dir is None else str(output_dir),
        },
        verbose_payload=payload,
        verbose_json=args.verbose_json,
    )
    if payload["status"] != "accepted":
        raise SystemExit(1)


def _build_payload(
    *,
    args: argparse.Namespace,
    controls: DMCRunControls,
    seeds: list[int],
    cases: list[TrappedCase],
    requested_workers: int,
) -> dict[str, Any]:
    initialization = InitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    collective_rn = (
        CollectiveRNControls(
            cadence_tau=args.collective_cadence_tau,
            proposal_family=args.proposal_family,
            target_family=args.target_family,
            component_log_scales=_parse_float_tuple(args.component_log_scales),
            component_probabilities=_parse_float_tuple(args.component_probabilities),
        )
        if args.collective_rn
        else None
    )
    pure_config = PureWalkingConfig(
        lag_steps=_parse_int_tuple(args.pure_fw_lags),
        density_lag_steps=(
            None
            if args.pure_fw_density_lags is None
            else _parse_int_tuple(args.pure_fw_density_lags)
        ),
        observables=_parse_str_tuple(args.pure_fw_observables),
        observable_source=args.pure_fw_observable_source,
        min_block_count=args.pure_fw_min_block_count,
        min_walker_weight_ess=args.pure_fw_min_walker_weight_ess,
        min_source_ancestor_ess=args.pure_fw_min_source_ancestor_ess,
        max_source_family_fraction=args.pure_fw_max_source_family_fraction,
        plateau_window_lag_count=args.pure_fw_plateau_window_lag_count,
        block_size_steps=1,
        collection_stride_steps=args.pure_fw_collection_stride_steps,
        density_collection_stride_steps=args.pure_fw_density_collection_stride_steps,
        density_plateau_relative_l2_tolerance=(args.density_plateau_relative_l2_tolerance),
        transport_invariant_tests_passed=("lag0_identity",),
    )
    tolerances = FiniteAN2ReferenceTolerances(
        energy_abs=args.energy_abs_tolerance,
        pure_r2_relative=args.pure_r2_relative_tolerance,
        pure_rms_relative=args.pure_rms_relative_tolerance,
        pure_density_l2=args.pure_density_l2_tolerance,
        density_accounting_abs=args.density_accounting_tolerance,
    )
    case_payloads: list[dict[str, Any]] = []
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds) * max(1, len(cases)),
        label=(
            "Finite-A N2 reference with collective RN"
            if collective_rn is not None
            else "Finite-A N2 reference"
        ),
        enabled=progress_requested(args.progress),
    ) as bar:
        for case in cases:
            case_payloads.append(
                summarize_finite_a_n2_reference_case(
                    case,
                    controls,
                    seeds,
                    pure_config=pure_config,
                    tolerances=tolerances,
                    reference_grid_points=args.reference_grid_points,
                    reference_y_max=args.reference_y_max,
                    parallel_workers=requested_workers,
                    progress=bar,
                    trace_output_dir=None,
                    ess_warning_fraction=args.ess_warning_fraction,
                    ess_invalid_fraction=args.ess_invalid_fraction,
                    log_weight_span_warning=args.log_weight_span_warning,
                    initialization=initialization,
                    collective_rn=collective_rn,
                    guide_family=args.guide_family,
                )
            )
    status = (
        "accepted"
        if all(case["status"] == "accepted" for case in case_payloads)
        else "one_or_more_reference_cases_unresolved"
    )
    return {
        "schema_version": "finite_a_n2_reference_packet_v2",
        "status": status,
        "validation": "finite-A DMC/FW against the deterministic N=2 reference",
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers_requested": requested_workers,
        "initialization_mode": args.initialization_mode,
        "init_width_log_sigma": args.init_width_log_sigma,
        "relative_alpha": args.relative_alpha,
        "breathing_preburn_steps": args.breathing_preburn_steps,
        "breathing_preburn_log_step": args.breathing_preburn_log_step,
        "collective_rn": (None if collective_rn is None else collective_rn.to_metadata()),
        "guide_family": args.guide_family,
        "reference_grid_points": args.reference_grid_points,
        "reference_y_max": args.reference_y_max,
        "pure_config": {
            "lag_steps": list(pure_config.lag_steps),
            "density_lag_steps": (
                None
                if pure_config.density_lag_steps is None
                else list(pure_config.density_lag_steps)
            ),
            "observables": list(pure_config.observables),
            "observable_source": pure_config.observable_source,
            "min_block_count": pure_config.min_block_count,
            "min_walker_weight_ess": pure_config.min_walker_weight_ess,
            "min_source_ancestor_ess": pure_config.min_source_ancestor_ess,
            "max_source_family_fraction": pure_config.max_source_family_fraction,
            "block_size_steps": pure_config.block_size_steps,
            "collection_stride_steps": pure_config.collection_stride_steps,
            "density_collection_stride_steps": (pure_config.density_collection_stride_steps),
            "density_plateau_relative_l2_tolerance": (
                pure_config.density_plateau_relative_l2_tolerance
            ),
        },
        "tolerances": tolerances.to_payload(),
        "plot_formats": list(_parse_str_tuple(args.plot_formats)),
        "case_table": [_case_table_row(case) for case in case_payloads],
        "case_results": case_payloads,
    }


def _case_table_row(case: dict[str, Any]) -> dict[str, Any]:
    comparison = case["comparison"]
    return {
        "case_id": case["case_id"],
        "status": case["status"],
        "benchmark_packet_status": case["benchmark_packet"]["status"],
        "energy_abs_error": comparison["energy"]["abs_error"],
        "pure_r2_relative_error": comparison["r2"]["pure_relative_error"],
        "pure_rms_relative_error": comparison["rms"]["pure_relative_error"],
        "pure_density_relative_l2": comparison["density"]["pure_relative_l2"],
        "density_accounting_abs_error": comparison["density"]["density_accounting_abs_error"],
    }


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(_parse_ints(value))


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(_parse_floats(value))


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one string")
    return values


def _format_float_tuple(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)


if __name__ == "__main__":
    main()
