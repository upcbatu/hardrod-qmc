from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io import progress_requested
from hrdmc.io.schema import to_jsonable
from hrdmc.plotting.figures.finite_a_n2_reference import (
    write_finite_a_n2_reference_plots,
)
from hrdmc.workflows.anchors.finite_a_n2 import (
    FiniteAN2ReferenceTolerances,
    summarize_finite_a_n2_reference_case,
    write_finite_a_n2_reference_artifacts,
    write_finite_a_n2_reference_manifest,
)
from hrdmc.workflows.dmc.rn_block import (
    DEFAULT_COMPONENT_LOG_SCALES,
    DEFAULT_COMPONENT_PROBABILITIES,
    DEFAULT_RN_GUIDE_FAMILY,
    DEFAULT_RN_PROPOSAL_FAMILY,
    DEFAULT_RN_TARGET_FAMILY,
    RN_GUIDE_FAMILIES,
    RN_PROPOSAL_FAMILIES,
    RN_TARGET_FAMILIES,
    RNCase,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
    resolve_parallel_workers,
    rn_progress_bar,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run N=2 finite-a trapped RN-DMC/FW packets against a deterministic "
            "relative-coordinate reference."
        )
    )
    parser.add_argument("--rod-length-values", default="0.5")
    parser.add_argument("--omega-values", default="0.2,0.1,0.05")
    parser.add_argument("--seeds", default="1001,1002")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=240.0)
    parser.add_argument("--store-every", type=int, default=40)
    parser.add_argument("--grid-extent", type=float, default=20.0)
    parser.add_argument("--n-bins", type=int, default=240)
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument(
        "--proposal-family",
        choices=RN_PROPOSAL_FAMILIES,
        default=DEFAULT_RN_PROPOSAL_FAMILY,
    )
    parser.add_argument(
        "--guide-family",
        choices=RN_GUIDE_FAMILIES,
        default=DEFAULT_RN_GUIDE_FAMILY,
    )
    parser.add_argument(
        "--target-family",
        choices=RN_TARGET_FAMILIES,
        default=DEFAULT_RN_TARGET_FAMILY,
    )
    parser.add_argument(
        "--component-log-scales",
        default=_format_float_tuple(DEFAULT_COMPONENT_LOG_SCALES),
    )
    parser.add_argument(
        "--component-probabilities",
        default=_format_float_tuple(DEFAULT_COMPONENT_PROBABILITIES),
    )
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-no-go-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument("--pure-fw-lags", default="0,10,20,30,40,50")
    parser.add_argument("--pure-fw-observables", default="r2,density")
    parser.add_argument(
        "--pure-fw-observable-source",
        choices=("raw_r2", "r2_rb"),
        default="raw_r2",
    )
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=1)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    controls = RNRunControls(
        dt=args.dt,
        walkers=args.walkers,
        tau_block=args.tau,
        rn_cadence_tau=args.rn_cadence,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=args.store_every,
        grid_extent=args.grid_extent,
        n_bins=args.n_bins,
    )
    seeds = _parse_ints(args.seeds)
    cases = [
        RNCase(n_particles=2, rod_length=rod_length, omega=omega)
        for rod_length in _parse_floats(args.rod_length_values)
        for omega in _parse_floats(args.omega_values)
    ]
    requested_workers = resolve_parallel_workers(len(seeds), args.parallel_workers)
    payload = _build_payload(
        args=args,
        controls=controls,
        seeds=seeds,
        cases=cases,
        requested_workers=requested_workers,
    )
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root,
            ArtifactRoute("dmc", "rn_block", "finite_a_n2_reference_packet"),
        )
        plot_paths: list[str] = []
        if not args.skip_plots:
            plot_paths = write_finite_a_n2_reference_plots(
                output_dir,
                payload,
                formats=_parse_str_tuple(args.plot_formats),
            )
            payload["plots"] = plot_paths
        artifacts = write_finite_a_n2_reference_artifacts(output_dir, payload)
        artifacts.extend(output_dir / path for path in plot_paths)
        write_finite_a_n2_reference_manifest(
            output_dir,
            payload,
            artifacts,
            controls,
            seeds,
            sys.argv,
        )
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))
    if payload["status"] != "FINITE_A_N2_REFERENCE_PACKET_GO":
        raise SystemExit(1)


def _build_payload(
    *,
    args: argparse.Namespace,
    controls: RNRunControls,
    seeds: list[int],
    cases: list[RNCase],
    requested_workers: int,
) -> dict[str, Any]:
    initialization = RNInitializationControls(
        mode=args.initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=args.breathing_preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
    )
    proposal = RNCollectiveProposalControls(
        component_log_scales=_parse_float_tuple(args.component_log_scales),
        component_probabilities=_parse_float_tuple(args.component_probabilities),
    )
    pure_config = PureWalkingConfig(
        lag_steps=_parse_int_tuple(args.pure_fw_lags),
        observables=_parse_str_tuple(args.pure_fw_observables),
        observable_source=args.pure_fw_observable_source,
        min_block_count=args.pure_fw_min_block_count,
        min_walker_weight_ess=args.pure_fw_min_walker_weight_ess,
        block_size_steps=1,
        collection_stride_steps=args.pure_fw_collection_stride_steps,
        density_plateau_relative_l2_tolerance=(
            args.density_plateau_relative_l2_tolerance
        ),
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
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds) * max(1, len(cases)),
        label="RN finite-a N2 reference packet",
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
                    ess_no_go_fraction=args.ess_no_go_fraction,
                    log_weight_span_warning=args.log_weight_span_warning,
                    initialization=initialization,
                    proposal=proposal,
                    proposal_family=args.proposal_family,
                    guide_family=args.guide_family,
                    target_family=args.target_family,
                )
            )
    status = (
        "FINITE_A_N2_REFERENCE_PACKET_GO"
        if all(case["status"] == "FINITE_A_N2_REFERENCE_PACKET_GO" for case in case_payloads)
        else "FINITE_A_N2_REFERENCE_PACKET_NO_GO"
    )
    return {
        "schema_version": "finite_a_n2_reference_packet_v1",
        "status": status,
        "benchmark_tier": "N=2 finite-a trapped deterministic reference packet",
        "claim_boundary": (
            "This packet validates the production finite-a RN-DMC/FW flow "
            "against the deterministic N=2 trapped hard-rod reference. It is "
            "a finite-a solver anchor, not an N>2 exact benchmark."
        ),
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers_requested": requested_workers,
        "initialization_mode": args.initialization_mode,
        "init_width_log_sigma": args.init_width_log_sigma,
        "breathing_preburn_steps": args.breathing_preburn_steps,
        "breathing_preburn_log_step": args.breathing_preburn_log_step,
        "proposal_family": args.proposal_family,
        "guide_family": args.guide_family,
        "target_family": args.target_family,
        **proposal.to_metadata(),
        "reference_grid_points": args.reference_grid_points,
        "reference_y_max": args.reference_y_max,
        "pure_config": {
            "lag_steps": list(pure_config.lag_steps),
            "observables": list(pure_config.observables),
            "observable_source": pure_config.observable_source,
            "min_block_count": pure_config.min_block_count,
            "min_walker_weight_ess": pure_config.min_walker_weight_ess,
            "block_size_steps": pure_config.block_size_steps,
            "collection_stride_steps": pure_config.collection_stride_steps,
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
        "density_accounting_abs_error": comparison["density"][
            "density_accounting_abs_error"
        ],
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
