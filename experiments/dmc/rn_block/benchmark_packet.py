from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io import progress_requested
from hrdmc.io.artifacts import build_run_provenance, write_json, write_run_manifest
from hrdmc.io.schema import to_jsonable
from hrdmc.plotting import write_benchmark_packet_plots
from hrdmc.workflows.dmc.benchmark_packet import (
    summarize_benchmark_packet_case,
    write_benchmark_packet_seed_table,
    write_benchmark_packet_table,
)
from hrdmc.workflows.dmc.rn_block import (
    RN_GUIDE_FAMILIES,
    RN_PROPOSAL_FAMILIES,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
    parse_case,
    parse_seeds,
    rn_progress_bar,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls

DEFAULT_CASE = "N8_a0.5_omega0.05"
DEFAULT_LAGS = "0,10,20,30,40,50,100,200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one RN-DMC benchmark packet with energy gates and transported FW."
    )
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seeds", default="1001,1002")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--rn-cadence", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=120.0)
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
        default="gap-h-transform",
    )
    parser.add_argument("--guide-family", choices=RN_GUIDE_FAMILIES, default="auto")
    parser.add_argument(
        "--component-log-scales",
        default="-0.015,-0.010,-0.004,0.000,0.004,0.010,0.015",
    )
    parser.add_argument(
        "--component-probabilities",
        default="0.03,0.10,0.22,0.30,0.22,0.10,0.03",
    )
    parser.add_argument("--ess-warning-fraction", type=float, default=0.20)
    parser.add_argument("--ess-no-go-fraction", type=float, default=0.10)
    parser.add_argument("--log-weight-span-warning", type=float, default=50.0)
    parser.add_argument("--pure-fw-lags", default=DEFAULT_LAGS)
    parser.add_argument(
        "--pure-fw-observables",
        default="r2,density",
    )
    parser.add_argument(
        "--pure-fw-observable-source",
        choices=("raw_r2", "r2_rb"),
        default="raw_r2",
    )
    parser.add_argument("--pure-fw-block-size-steps", type=int, default=1)
    parser.add_argument("--pure-fw-min-block-count", type=int, default=30)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument(
        "--pure-fw-pair-max",
        type=float,
        default=None,
        help="Pair-distance histogram max; defaults to 2*grid_extent.",
    )
    parser.add_argument("--pure-fw-pair-bins", type=int, default=240)
    parser.add_argument(
        "--pure-fw-k-values",
        default="0.05,0.1,0.2,0.4,0.8,1.6",
        help="Comma-separated k values for finite-cloud structure factor.",
    )
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-formats", default="png,pdf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    case = parse_case(args.case)
    seeds = parse_seeds(args.seeds)
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
        pair_bin_edges=_pair_edges(args),
        structure_k_values=_parse_float_array(args.pure_fw_k_values),
        min_block_count=args.pure_fw_min_block_count,
        min_walker_weight_ess=args.pure_fw_min_walker_weight_ess,
        block_size_steps=args.pure_fw_block_size_steps,
        transport_invariant_tests_passed=(
            "lag0_identity",
            "deterministic_parent_map",
            "weight_gauge_shift_cancellation",
        ),
    )
    output_dir = args.output_dir or artifact_dir(
        repo_root,
        ArtifactRoute("dmc", "rn_block", "benchmark_packet"),
    )
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label="RN benchmark packet",
        enabled=progress_requested(args.progress),
    ) as bar:
        payload = summarize_benchmark_packet_case(
            case,
            controls,
            seeds,
            pure_config=pure_config,
            parallel_workers=args.parallel_workers,
            progress=bar,
            trace_output_dir=None if args.skip_write else output_dir,
            ess_warning_fraction=args.ess_warning_fraction,
            ess_no_go_fraction=args.ess_no_go_fraction,
            log_weight_span_warning=args.log_weight_span_warning,
            initialization=initialization,
            proposal=proposal,
            proposal_family=args.proposal_family,
            guide_family=args.guide_family,
        )
    if not args.skip_write:
        plot_paths: list[str] = []
        if not args.skip_plots:
            plot_paths = write_benchmark_packet_plots(
                output_dir,
                payload,
                formats=_parse_str_tuple(args.plot_formats),
            )
            payload["plots"] = plot_paths
        summary_path = output_dir / "summary.json"
        write_json(summary_path, payload)
        seed_table = write_benchmark_packet_seed_table(output_dir, payload["seed_results"])
        packet_table = write_benchmark_packet_table(output_dir, payload)
        artifacts = [summary_path, seed_table, packet_table]
        artifacts.extend(output_dir / path for path in plot_paths)
        write_run_manifest(
            output_dir,
            run_name="rn_block_benchmark_packet",
            config={
                "case": case.case_id,
                "seeds": seeds,
                "controls": controls_to_dict(controls),
                "parallel_workers": args.parallel_workers,
                "initialization_mode": args.initialization_mode,
                "init_width_log_sigma": args.init_width_log_sigma,
                "breathing_preburn_steps": args.breathing_preburn_steps,
                "breathing_preburn_log_step": args.breathing_preburn_log_step,
                "component_log_scales": list(proposal.component_log_scales),
                "component_probabilities": list(proposal.component_probabilities),
                "proposal_family": args.proposal_family,
                "guide_family": args.guide_family,
                "pure_config": payload["pure_config"],
                "plot_formats": list(_parse_str_tuple(args.plot_formats)),
            },
            artifacts=artifacts,
            schema_version="rn_block_benchmark_packet_v1",
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


def _parse_float_array(value: str) -> np.ndarray:
    return np.asarray(_parse_float_tuple(value), dtype=float)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one integer value is required")
    return values


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one observable is required")
    return values


def _pair_edges(args: argparse.Namespace) -> np.ndarray | None:
    observables = set(_parse_str_tuple(args.pure_fw_observables))
    if "pair_distance_density" not in observables:
        return None
    pair_max = float(
        2.0 * args.grid_extent
        if args.pure_fw_pair_max is None
        else args.pure_fw_pair_max
    )
    if pair_max <= 0.0:
        raise ValueError("pure-fw-pair-max must be positive")
    if args.pure_fw_pair_bins <= 0:
        raise ValueError("pure-fw-pair-bins must be positive")
    return np.linspace(0.0, pair_max, args.pure_fw_pair_bins + 1)


if __name__ == "__main__":
    main()
