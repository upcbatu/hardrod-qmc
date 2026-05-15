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
from hrdmc.workflows.dmc.pure_walking import (
    summarize_pure_walking_case,
    write_pure_walking_seed_table,
)
from hrdmc.workflows.dmc.rn_block import (
    DEFAULT_RN_TARGET_FAMILY,
    RN_GUIDE_FAMILIES,
    RN_PROPOSAL_FAMILIES,
    RN_TARGET_FAMILIES,
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
SUPPORTED_OBSERVABLES = {"r2", "density"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run transported auxiliary forward-walking on RN-block DMC transport events."
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
    parser.add_argument(
        "--guide-family",
        choices=RN_GUIDE_FAMILIES,
        default="auto",
    )
    parser.add_argument(
        "--target-family",
        choices=RN_TARGET_FAMILIES,
        default=DEFAULT_RN_TARGET_FAMILY,
    )
    parser.add_argument(
        "--component-log-scales",
        default="-0.015,-0.010,-0.004,0.000,0.004,0.010,0.015",
    )
    parser.add_argument(
        "--component-probabilities",
        default="0.03,0.10,0.22,0.30,0.22,0.10,0.03",
    )
    parser.add_argument("--lags", default=DEFAULT_LAGS)
    parser.add_argument(
        "--observables",
        default="r2",
        help="Comma-separated transported FW observables. Supported: r2,density.",
    )
    parser.add_argument("--observable-source", choices=("raw_r2", "r2_rb"), default="raw_r2")
    parser.add_argument("--block-size-steps", type=int, default=1)
    parser.add_argument("--collection-stride-steps", type=int, default=1)
    parser.add_argument("--min-block-count", type=int, default=30)
    parser.add_argument("--min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--plateau-window-lag-count", type=int, default=4)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = repo_root_from(Path(__file__))
    case = parse_case(args.case)
    seeds = parse_seeds(args.seeds)
    observables = _parse_str_tuple(args.observables)
    unsupported_observables = set(observables) - SUPPORTED_OBSERVABLES
    if unsupported_observables:
        parser.error(
            "unsupported --observables for this runner: "
            + ",".join(sorted(unsupported_observables))
            + "; supported values are r2,density"
        )
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
        lag_steps=_parse_int_tuple(args.lags),
        observables=observables,
        observable_source=args.observable_source,
        density_bin_edges=(
            np.linspace(-args.grid_extent, args.grid_extent, args.n_bins + 1)
            if "density" in observables
            else None
        ),
        min_block_count=args.min_block_count,
        min_walker_weight_ess=args.min_walker_weight_ess,
        plateau_window_lag_count=args.plateau_window_lag_count,
        block_size_steps=args.block_size_steps,
        collection_stride_steps=args.collection_stride_steps,
        transport_invariant_tests_passed=(
            "lag0_identity",
            "deterministic_parent_map",
            "weight_gauge_shift_cancellation",
        ),
    )
    output_dir = args.output_dir or artifact_dir(
        repo_root, ArtifactRoute("dmc", "rn_block", "pure_walking_anchor")
    )
    with rn_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label="RN pure walking",
        enabled=progress_requested(args.progress),
    ) as bar:
        payload = summarize_pure_walking_case(
            case,
            controls,
            seeds,
            pure_config=pure_config,
            parallel_workers=args.parallel_workers,
            progress=bar,
            initialization=initialization,
            proposal=proposal,
            proposal_family=args.proposal_family,
            guide_family=args.guide_family,
            target_family=args.target_family,
        )
    payload["claim_boundary"] = (
        "transported auxiliary FW candidate pure estimator; paper coordinate "
        "claims require plateau, effective-sample, density-accounting, and "
        "population checks"
    )
    if not args.skip_write:
        summary_path = output_dir / "summary.json"
        write_json(summary_path, payload)
        seed_table = write_pure_walking_seed_table(output_dir, payload["seed_results"])
        write_run_manifest(
            output_dir,
            run_name="rn_block_pure_walking_anchor",
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
                "target_family": args.target_family,
                "pure_config": payload["pure_config"],
            },
            artifacts=[summary_path, seed_table],
            schema_version="transported_pure_walking_case_v1",
            provenance=build_run_provenance(sys.argv),
        )
    print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("at least one numeric value is required")
    return values


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


if __name__ == "__main__":
    main()
