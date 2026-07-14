from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io import print_run_summary, progress_requested
from hrdmc.theory.units import HO_TRAP_OMEGA
from hrdmc.workflows.anchors.exact_validation import (
    HomogeneousRingAnchor,
    TrappedTGAnchor,
    anchor_row_from_homogeneous,
    anchor_row_from_trapped,
    run_homogeneous_ring_anchor,
    run_trapped_tg_anchor,
    write_exact_validation_manifest,
    write_packet_artifacts,
)
from hrdmc.workflows.dmc.trapped import (
    DMCRunControls,
    controls_to_dict,
    dmc_progress_bar,
    resolve_parallel_workers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact trapped and homogeneous validation cases."
    )
    parser.add_argument("--trapped-n-values", default="2,4")
    parser.add_argument("--homogeneous-n-values", default="4,8")
    parser.add_argument("--homogeneous-eta-values", default="0.1,0.5")
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--seeds", default="301,302")
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument(
        "--local-step-method",
        choices=("euler", "metropolis"),
        default="metropolis",
    )
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument(
        "--relative-alpha",
        type=float,
        default=None,
        help=(
            "Optional reduced-coordinate internal Gaussian width for trapped "
            "TG guide anchors. The center-of-mass width remains harmonic."
        ),
    )
    parser.add_argument("--burn-tau", type=float, default=20.0)
    parser.add_argument("--production-tau", type=float, default=40.0)
    parser.add_argument("--store-every", type=int, default=20)
    parser.add_argument("--grid-extent", type=float, default=12.0)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=800,
        help=(
            "Density histogram bins. The default is intentionally high enough "
            "to resolve finite-N trapped density peaks in report figures."
        ),
    )
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--energy-tolerance", type=float, default=1e-8)
    parser.add_argument("--homogeneous-samples", type=int, default=6)
    parser.add_argument("--homogeneous-tolerance", type=float, default=1e-7)
    parser.add_argument("--homogeneous-seed", type=int, default=20260511)
    parser.add_argument("--pure-fw-lags", default="0,10,20,30")
    parser.add_argument("--pure-fw-density-lags", default=None)
    parser.add_argument("--pure-fw-observables", default="r2,density")
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-min-source-ancestor-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-max-source-family-fraction", type=float, default=0.25)
    parser.add_argument("--pure-fw-plateau-window-lag-count", type=int, default=4)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=1)
    parser.add_argument(
        "--pure-fw-density-collection-stride-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--density-plateau-relative-l2-tolerance",
        type=float,
        default=0.03,
    )
    parser.add_argument("--pure-r2-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--pure-rms-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--pure-density-l2-tolerance", type=float, default=0.10)
    parser.add_argument("--density-accounting-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--density-shape-min-bins", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-formats", default="png,pdf")
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
    trapped_anchors = _trapped_anchors(args.trapped_n_values)
    homogeneous_anchors = _homogeneous_anchors(
        args.homogeneous_n_values,
        args.homogeneous_eta_values,
    )
    requested_workers = resolve_parallel_workers(len(seeds), args.parallel_workers)
    payload = _build_payload(
        args=args,
        controls=controls,
        seeds=seeds,
        trapped_anchors=trapped_anchors,
        homogeneous_anchors=homogeneous_anchors,
        requested_workers=requested_workers,
    )
    output_dir: Path | None = None
    if not args.no_write:
        write_output_dir = Path(
            args.output_dir
            or artifact_dir(
                repo_root,
                ArtifactRoute("dmc", "local", "exact_validation_packet"),
            )
        )
        output_dir = write_output_dir
        paths = write_packet_artifacts(
            write_output_dir,
            payload,
            _parse_str_tuple(args.plot_formats),
        )
        write_exact_validation_manifest(write_output_dir, payload, paths, controls, seeds, sys.argv)
    print_run_summary(
        run="exact_validation_packet",
        status=str(payload["status"]),
        summary={
            "anchor_count": len(payload["anchor_table"]),
            "trapped_anchor_count": len(trapped_anchors),
            "homogeneous_anchor_count": len(homogeneous_anchors),
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
    trapped_anchors: list[TrappedTGAnchor],
    homogeneous_anchors: list[HomogeneousRingAnchor],
    requested_workers: int,
) -> dict[str, Any]:
    trapped_payloads: list[dict[str, Any]] = []
    trapped_rows: list[dict[str, Any]] = []
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds) * max(1, len(trapped_anchors)),
        label="Exact validation packet",
        enabled=progress_requested(args.progress),
    ) as bar:
        for anchor in trapped_anchors:
            payload = run_trapped_tg_anchor(
                anchor,
                controls,
                seeds,
                worker_count=requested_workers,
                energy_tolerance=args.energy_tolerance,
                pure_lag_steps=tuple(_parse_ints(args.pure_fw_lags)),
                pure_density_lag_steps=(
                    None
                    if args.pure_fw_density_lags is None
                    else tuple(_parse_ints(args.pure_fw_density_lags))
                ),
                pure_observables=tuple(_parse_str_tuple(args.pure_fw_observables)),
                pure_min_block_count=args.pure_fw_min_block_count,
                pure_min_walker_weight_ess=args.pure_fw_min_walker_weight_ess,
                pure_min_source_ancestor_ess=(args.pure_fw_min_source_ancestor_ess),
                pure_max_source_family_fraction=(args.pure_fw_max_source_family_fraction),
                pure_plateau_window_lag_count=(args.pure_fw_plateau_window_lag_count),
                pure_collection_stride_steps=args.pure_fw_collection_stride_steps,
                pure_density_collection_stride_steps=(args.pure_fw_density_collection_stride_steps),
                density_plateau_relative_l2_tolerance=(args.density_plateau_relative_l2_tolerance),
                pure_r2_relative_tolerance=args.pure_r2_relative_tolerance,
                pure_rms_relative_tolerance=args.pure_rms_relative_tolerance,
                pure_density_l2_tolerance=args.pure_density_l2_tolerance,
                density_accounting_tolerance=args.density_accounting_tolerance,
                density_shape_min_bins=args.density_shape_min_bins,
                progress=bar,
            )
            trapped_payloads.append(payload)
            trapped_rows.append(anchor_row_from_trapped(payload))
    homogeneous_payloads = [
        run_homogeneous_ring_anchor(
            anchor,
            rod_length=args.rod_length,
            samples_per_case=args.homogeneous_samples,
            seed=args.homogeneous_seed + index,
            tolerance=args.homogeneous_tolerance,
        )
        for index, anchor in enumerate(homogeneous_anchors)
    ]
    homogeneous_rows = [anchor_row_from_homogeneous(row) for row in homogeneous_payloads]
    anchor_table = [*trapped_rows, *homogeneous_rows]
    status = (
        "accepted"
        if all(bool(row["accepted"]) for row in anchor_table)
        else "one_or_more_exact_references_unresolved"
    )
    return {
        "schema_version": "dmc_exact_validation_packet_v3",
        "status": status,
        "validation": "exact trapped TG and homogeneous hard-rod cases",
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "parallel_workers_requested": requested_workers,
        "trapped_tg_anchors": trapped_payloads,
        "homogeneous_ring_anchors": homogeneous_payloads,
        "anchor_table": anchor_table,
    }


def _trapped_anchors(n_values: str) -> list[TrappedTGAnchor]:
    return [TrappedTGAnchor(n_particles=n, omega=HO_TRAP_OMEGA) for n in _parse_ints(n_values)]


def _homogeneous_anchors(
    n_values: str,
    eta_values: str,
) -> list[HomogeneousRingAnchor]:
    return [
        HomogeneousRingAnchor(n_particles=n, packing_fraction=eta)
        for n in _parse_ints(n_values)
        for eta in _parse_floats(eta_values)
    ]


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one output format")
    return values


if __name__ == "__main__":
    main()
