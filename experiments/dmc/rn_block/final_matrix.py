from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts import repo_root_from
from hrdmc.io.artifacts import (
    config_fingerprint,
    ensure_dir,
    verify_run_manifest,
    write_json,
)
from hrdmc.workflows.dmc.rn_block import (
    RNRunControls,
    controls_to_dict,
    make_grid,
    parse_case,
    parse_seeds,
)

DEFAULT_CASES = "N10_A0.1,N10_A1,N10_A10,N20_A0.1,N20_A1,N20_A10"
DEFAULT_SEEDS = "7001,7002,7003"
# The finite-rod N=2 anchor requires a physical forward-walking projection
# time well beyond the earlier 7/omega ladder.  At dt=0.0025 this reaches
# 50/omega while retaining ample production support for tau_prod=480.
DEFAULT_LAGS = "0,2000,4000,8000,12000,16000,20000"
DEFAULT_DENSITY_LAGS = "0,8000,12000,20000"
DEFAULT_DENSITY_COLLECTION_STRIDE_STEPS = 4000
DEFAULT_RN_CADENCE = 0.01
FW_LAG_TIMES = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
DENSITY_FW_LAG_TIMES = (0.0, 20.0, 30.0, 50.0)
STORE_INTERVAL_TAU = 0.025
FW_COLLECTION_INTERVAL_TAU = 0.05
DENSITY_FW_COLLECTION_INTERVAL_TAU = 10.0


@dataclass(frozen=True)
class RowMethod:
    dt: float
    relative_alpha: float | None
    initialization_mode: str
    init_width_log_sigma: float
    breathing_preburn_steps: int
    breathing_preburn_log_step: float
    store_every: int
    pure_fw_lags: tuple[int, ...]
    pure_fw_density_lags: tuple[int, ...]
    pure_fw_collection_stride_steps: int
    pure_fw_density_collection_stride_steps: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final trapped hard-rod DMC matrix by dispatching "
            "benchmark-packet cases with shared settings."
        )
    )
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument(
        "--local-step-method",
        choices=("euler", "metropolis"),
        default="metropolis",
    )
    parser.add_argument("--walkers", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--burn-tau", type=float, default=60.0)
    parser.add_argument("--production-tau", type=float, default=480.0)
    parser.add_argument("--store-every", type=int, default=10)
    parser.add_argument(
        "--grid-extent",
        type=float,
        default=35.0,
        help="Minimum half-width of the density grid in oscillator units.",
    )
    parser.add_argument(
        "--excluded-volume-margin",
        type=float,
        default=35.0,
        help=(
            "Extra half-width reserved beyond N*a/2 for finite-rod cases. "
            "The final LDA grid may grow further when the cloud requires it."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=840)
    parser.add_argument(
        "--max-density-bin-width",
        type=float,
        default=0.20,
        help="Largest density-bin width in oscillator units after grid planning.",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("tight-lattice", "lda-rms-lattice", "lda-rms-logspread"),
        default="lda-rms-logspread",
    )
    parser.add_argument("--init-width-log-sigma", type=float, default=0.10)
    parser.add_argument("--breathing-preburn-steps", type=int, default=1000)
    parser.add_argument("--breathing-preburn-log-step", type=float, default=0.04)
    parser.add_argument("--pure-fw-lags", default=DEFAULT_LAGS)
    parser.add_argument("--pure-fw-density-lags", default=DEFAULT_DENSITY_LAGS)
    parser.add_argument("--pure-fw-block-size-steps", type=int, default=1)
    parser.add_argument("--pure-fw-collection-stride-steps", type=int, default=20)
    parser.add_argument(
        "--pure-fw-density-collection-stride-steps",
        type=int,
        default=DEFAULT_DENSITY_COLLECTION_STRIDE_STEPS,
    )
    parser.add_argument("--pure-fw-min-block-count", type=int, default=20)
    parser.add_argument("--pure-fw-min-walker-weight-ess", type=float, default=30.0)
    parser.add_argument("--pure-fw-density-plateau-window-lag-count", type=int, default=3)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--plot-formats", default="png,pdf")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/dmc/final_matrix/local_dmc_dt0025_split_fw50"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(Path(__file__))
    cases = _parse_cases(args.cases)
    seeds = parse_seeds(args.seeds)
    output_root = ensure_dir(args.output_root)

    records: list[dict[str, Any]] = []
    for case_id in cases:
        case_output_dir = output_root / case_id
        method = _row_method(args, case_id)
        grid_plan = _case_grid_plan(args, case_id)
        record: dict[str, Any] = {
            "case": case_id,
            "output_dir": str(case_output_dir),
            "summary": str(case_output_dir / "summary.json"),
            "grid_plan": grid_plan,
            "method": _row_method_metadata(method),
        }
        summary_path = case_output_dir / "summary.json"
        completed, completion_errors = _verified_completed_row(
            args,
            case_id,
            case_output_dir,
            grid_plan,
            method,
        )
        if completed and not args.force:
            record["status"] = "skipped_verified_complete"
            record["grid_plan"] = _existing_grid_plan(summary_path, grid_plan)
            record["run_manifest"] = str(case_output_dir / "run_manifest.json")
            record["rerun_command"] = _benchmark_command(
                args,
                case_id,
                case_output_dir,
                grid_plan,
                method,
            )
            records.append(record)
            continue
        if completion_errors:
            record["existing_artifact_errors"] = completion_errors
        command = _benchmark_command(args, case_id, case_output_dir, grid_plan, method)
        record["command"] = command
        if args.dry_run:
            record["status"] = "planned"
            records.append(record)
            continue
        ensure_dir(case_output_dir)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=_subprocess_env(repo_root),
            check=False,
        )
        record["returncode"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        records.append(record)
        _write_matrix_manifest(output_root, args, cases, seeds, records)
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)

    manifest_path = _write_matrix_manifest(output_root, args, cases, seeds, records)
    print(json.dumps({"manifest": str(manifest_path), "rows": records}, indent=2))


def _parse_cases(value: str) -> list[str]:
    cases = [item.strip() for item in value.split(",") if item.strip()]
    if not cases:
        raise ValueError("at least one case is required")
    for case_id in cases:
        parse_case(case_id)
    return cases


def _case_grid_plan(args: argparse.Namespace, case_id: str) -> dict[str, float | int]:
    """Plan a finite-support density grid before dispatching one matrix row."""
    case = parse_case(case_id)
    minimum_extent = 0.5 * case.n_particles * case.rod_length
    requested_extent = max(args.grid_extent, minimum_extent + args.excluded_volume_margin)
    controls = _disabled_rn_controls(
        args,
        _row_method(args, case_id),
        grid_extent=requested_extent,
        n_bins=args.n_bins,
    )
    grid = make_grid(controls, case)
    planned_extent = _command_float(float(max(abs(grid[0]), abs(grid[-1]))))
    planned_bins = max(
        args.n_bins,
        math.ceil((2.0 * planned_extent) / args.max_density_bin_width) + 1,
    )
    return {
        "minimum_excluded_volume_extent": minimum_extent,
        "requested_grid_extent": requested_extent,
        "grid_extent": planned_extent,
        "n_bins": planned_bins,
        "density_bin_width": (2.0 * planned_extent) / (planned_bins - 1),
    }


def _existing_grid_plan(
    summary_path: Path,
    planned_grid: dict[str, float | int],
) -> dict[str, float | int]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    controls = payload.get("controls", {})
    grid_extent = controls.get("grid_extent")
    n_bins = controls.get("n_bins")
    if not isinstance(grid_extent, (int, float)) or not isinstance(n_bins, int):
        return planned_grid
    density = payload.get("paper_values", {}).get("density", {})
    density_x = density.get("x") if isinstance(density, dict) else None
    if isinstance(density_x, list) and len(density_x) >= 2:
        actual_extent = float(max(abs(density_x[0]), abs(density_x[-1])))
        actual_bins = len(density_x)
        return {
            **planned_grid,
            "configured_grid_extent": float(grid_extent),
            "grid_extent": actual_extent,
            "n_bins": actual_bins,
            "density_bin_width": float(density_x[1] - density_x[0]),
        }
    return {
        **planned_grid,
        "grid_extent": float(grid_extent),
        "n_bins": n_bins,
        "density_bin_width": (2.0 * float(grid_extent)) / (n_bins - 1),
    }


def _benchmark_command(
    args: argparse.Namespace,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> list[str]:
    command = [
        sys.executable,
        "experiments/dmc/rn_block/benchmark_packet.py",
        "--case",
        case_id,
        "--seeds",
        args.seeds,
        "--dt",
        _format_number(method.dt),
        "--walkers",
        str(args.walkers),
        "--local-step-method",
        args.local_step_method,
        "--tau",
        _format_number(args.tau),
        "--rn-cadence",
        _format_number(DEFAULT_RN_CADENCE),
        "--burn-tau",
        _format_number(args.burn_tau),
        "--production-tau",
        _format_number(args.production_tau),
        "--store-every",
        str(method.store_every),
        "--grid-extent",
        _format_number(float(grid_plan["grid_extent"])),
        "--n-bins",
        str(grid_plan["n_bins"]),
        "--initialization-mode",
        method.initialization_mode,
        "--init-width-log-sigma",
        _format_number(method.init_width_log_sigma),
        "--breathing-preburn-steps",
        str(method.breathing_preburn_steps),
        "--breathing-preburn-log-step",
        _format_number(method.breathing_preburn_log_step),
        "--proposal-family",
        "harmonic-mehler",
        "--guide-family",
        "reduced-tg",
        "--target-family",
        "primitive",
        "--component-log-scales",
        "0.0",
        "--component-probabilities",
        "1.0",
        "--pure-fw-lags",
        _format_int_tuple(method.pure_fw_lags),
        "--pure-fw-density-lags",
        _format_int_tuple(method.pure_fw_density_lags),
        "--pure-fw-block-size-steps",
        str(args.pure_fw_block_size_steps),
        "--pure-fw-collection-stride-steps",
        str(method.pure_fw_collection_stride_steps),
        "--pure-fw-density-collection-stride-steps",
        str(method.pure_fw_density_collection_stride_steps),
        "--pure-fw-min-block-count",
        str(args.pure_fw_min_block_count),
        "--pure-fw-min-walker-weight-ess",
        _format_number(args.pure_fw_min_walker_weight_ess),
        "--pure-fw-density-plateau-window-lag-count",
        str(args.pure_fw_density_plateau_window_lag_count),
        "--parallel-workers",
        str(args.parallel_workers),
        "--plot-formats",
        args.plot_formats,
        "--output-dir",
        str(output_dir),
        "--disable-rn",
    ]
    if method.relative_alpha is not None:
        command.extend(("--relative-alpha", _format_number(method.relative_alpha)))
    if args.progress:
        command.append("--progress")
    return command


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _write_matrix_manifest(
    output_root: Path,
    args: argparse.Namespace,
    cases: list[str],
    seeds: list[int],
    records: list[dict[str, Any]],
) -> Path:
    path = output_root / "final_matrix_manifest.json"
    existing_records = _existing_matrix_records(path)
    discovered_records = _discover_completed_rows(output_root, args)
    merged_records = {
        record["case"]: record for record in [*existing_records, *discovered_records, *records]
    }
    write_json(
        path,
        {
            "schema_version": "hardrod_final_matrix_v2",
            "requested_cases": cases,
            "seeds": seeds,
            "invocation_settings": {
                "dt": args.dt,
                "local_step_method": args.local_step_method,
                "walkers": args.walkers,
                "tau": args.tau,
                "burn_tau": args.burn_tau,
                "production_tau": args.production_tau,
                "store_every": args.store_every,
                "grid_extent": args.grid_extent,
                "excluded_volume_margin": args.excluded_volume_margin,
                "n_bins": args.n_bins,
                "max_density_bin_width": args.max_density_bin_width,
                "initialization_mode": args.initialization_mode,
                "init_width_log_sigma": args.init_width_log_sigma,
                "breathing_preburn_steps": args.breathing_preburn_steps,
                "breathing_preburn_log_step": args.breathing_preburn_log_step,
                "pure_fw_lags": args.pure_fw_lags,
                "pure_fw_density_lags": args.pure_fw_density_lags,
                "pure_fw_block_size_steps": args.pure_fw_block_size_steps,
                "pure_fw_collection_stride_steps": args.pure_fw_collection_stride_steps,
                "pure_fw_density_collection_stride_steps": (
                    args.pure_fw_density_collection_stride_steps
                ),
                "pure_fw_min_block_count": args.pure_fw_min_block_count,
                "pure_fw_min_walker_weight_ess": args.pure_fw_min_walker_weight_ess,
                "pure_fw_density_plateau_window_lag_count": (
                    args.pure_fw_density_plateau_window_lag_count
                ),
                "parallel_workers": args.parallel_workers,
                "plot_formats": args.plot_formats,
                "calculation": "metropolis_corrected_drift_diffusion_dmc",
                "row_method_policy": {
                    "a_over_aho_0p1_or_smaller": "dt=0.0025, reduced-TG default guide",
                    "a_over_aho_1": "dt=0.00125, relative-alpha=1.5",
                    "a_over_aho_10": "dt=0.00025, reduced-TG default guide",
                },
            },
            "rows": [merged_records[case_id] for case_id in sorted(merged_records)],
        },
    )
    return path


def _existing_matrix_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("rows", payload.get("records", []))
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("status") == "planned" and "case" in record
    ]


def _discover_completed_rows(output_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("N*_A*/summary.json")):
        case_id = summary_path.parent.name
        try:
            parse_case(case_id)
        except ValueError:
            continue
        grid_plan = _case_grid_plan(args, case_id)
        method = _row_method(args, case_id)
        completed, _errors = _verified_completed_row(
            args,
            case_id,
            summary_path.parent,
            grid_plan,
            method,
        )
        if not completed:
            continue
        records.append(
            {
                "case": case_id,
                "output_dir": str(summary_path.parent),
                "summary": str(summary_path),
                "run_manifest": str(summary_path.parent / "run_manifest.json"),
                "status": "verified_existing",
                "grid_plan": _existing_grid_plan(summary_path, grid_plan),
            }
        )
    return records


def _disabled_rn_controls(
    args: argparse.Namespace,
    method: RowMethod,
    *,
    grid_extent: float,
    n_bins: int,
) -> RNRunControls:
    return RNRunControls(
        dt=method.dt,
        walkers=args.walkers,
        tau_block=args.tau,
        rn_cadence_tau=DEFAULT_RN_CADENCE,
        collective_rn_enabled=False,
        burn_tau=args.burn_tau,
        production_tau=args.production_tau,
        store_every=method.store_every,
        grid_extent=grid_extent,
        n_bins=n_bins,
        local_step_method=args.local_step_method,
        relative_alpha=method.relative_alpha,
    )


def _verified_completed_row(
    args: argparse.Namespace,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> tuple[bool, list[str]]:
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False, ["missing summary.json or run_manifest.json"]
    verified, errors = verify_run_manifest(manifest_path)
    if not verified:
        return False, errors
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not isinstance(config, dict):
        return False, ["run manifest has no configuration payload"]
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        return False, ["run manifest configuration fingerprint mismatch"]
    expected = _expected_manifest_fields(args, case_id, grid_plan, method)
    mismatches = [key for key, value in expected.items() if _nested_value(config, key) != value]
    if mismatches:
        return False, [f"configuration mismatch: {', '.join(mismatches)}"]
    required = {
        "summary.json",
        "seed_table.csv",
        "packet_table.csv",
        "fw_plateau_table.csv",
        "energy_stationarity_table.csv",
        "density_fw_table.csv",
    }
    artifact_paths = {
        str(entry.get("path")) for entry in manifest.get("artifacts", []) if isinstance(entry, dict)
    }
    missing = sorted(required - artifact_paths)
    if missing:
        return False, [f"manifest missing required artifacts: {', '.join(missing)}"]
    return True, []


def _expected_manifest_fields(
    args: argparse.Namespace,
    case_id: str,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> dict[str, Any]:
    controls = _disabled_rn_controls(
        args,
        method,
        grid_extent=float(grid_plan["grid_extent"]),
        n_bins=int(grid_plan["n_bins"]),
    )
    return {
        "case": case_id,
        "seeds": parse_seeds(args.seeds),
        "controls": controls_to_dict(controls),
        "parallel_workers": args.parallel_workers,
        "disable_rn": True,
        "initialization_mode": method.initialization_mode,
        "init_width_log_sigma": method.init_width_log_sigma,
        "breathing_preburn_steps": method.breathing_preburn_steps,
        "breathing_preburn_log_step": method.breathing_preburn_log_step,
        "component_log_scales": [0.0],
        "component_probabilities": [1.0],
        "proposal_family": "harmonic-mehler",
        "guide_family": "reduced-tg",
        "target_family": "primitive",
        "pure_config.lag_steps": list(method.pure_fw_lags),
        "pure_config.density_lag_steps": list(method.pure_fw_density_lags),
        "pure_config.observables": ["r2", "density"],
        "pure_config.observable_source": "raw_r2",
        "pure_config.block_size_steps": args.pure_fw_block_size_steps,
        "pure_config.collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_config.density_collection_stride_steps": (
            method.pure_fw_density_collection_stride_steps
        ),
        "pure_config.min_block_count": args.pure_fw_min_block_count,
        "pure_config.min_walker_weight_ess": args.pure_fw_min_walker_weight_ess,
        "pure_config.plateau_window_lag_count": 4,
        "pure_config.density_plateau_window_lag_count": (
            args.pure_fw_density_plateau_window_lag_count
        ),
        "plot_formats": [value.strip() for value in args.plot_formats.split(",")],
    }


def _nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format_number(value: float) -> str:
    return f"{value:g}"


def _format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _steps_for_tau(tau: float, dt: float) -> int:
    if tau == 0.0:
        return 0
    steps = int(round(tau / dt))
    if steps <= 0 or not math.isclose(steps * dt, tau, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{tau=} is not representable at {dt=}")
    return steps


def _row_method(args: argparse.Namespace, case_id: str) -> RowMethod:
    case = parse_case(case_id)
    if math.isclose(case.rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.00025
        relative_alpha = 1.0
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
    elif math.isclose(case.rod_length, 1.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.00125
        relative_alpha = 1.5
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
    else:
        dt = args.dt
        relative_alpha = None
        initialization_mode = args.initialization_mode
        preburn_steps = args.breathing_preburn_steps
    return RowMethod(
        dt=dt,
        relative_alpha=relative_alpha,
        initialization_mode=initialization_mode,
        init_width_log_sigma=args.init_width_log_sigma,
        breathing_preburn_steps=preburn_steps,
        breathing_preburn_log_step=args.breathing_preburn_log_step,
        store_every=_steps_for_tau(STORE_INTERVAL_TAU, dt),
        pure_fw_lags=tuple(_steps_for_tau(tau, dt) for tau in FW_LAG_TIMES),
        pure_fw_density_lags=tuple(_steps_for_tau(tau, dt) for tau in DENSITY_FW_LAG_TIMES),
        pure_fw_collection_stride_steps=_steps_for_tau(FW_COLLECTION_INTERVAL_TAU, dt),
        pure_fw_density_collection_stride_steps=_steps_for_tau(
            DENSITY_FW_COLLECTION_INTERVAL_TAU,
            dt,
        ),
    )


def _row_method_metadata(method: RowMethod) -> dict[str, float | int | list[int] | None | str]:
    return {
        "dt": method.dt,
        "relative_alpha": method.relative_alpha,
        "initialization_mode": method.initialization_mode,
        "store_every": method.store_every,
        "pure_fw_lags": list(method.pure_fw_lags),
        "pure_fw_density_lags": list(method.pure_fw_density_lags),
        "pure_fw_collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_fw_density_collection_stride_steps": method.pure_fw_density_collection_stride_steps,
    }


def _command_float(value: float) -> float:
    """Return the float reconstructed from the command-line representation."""

    return float(_format_number(value))


if __name__ == "__main__":
    main()
