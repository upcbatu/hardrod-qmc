from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts import (
    config_fingerprint,
    ensure_dir,
    implementation_identity,
    verify_run_manifest,
    write_json,
)
from hrdmc.workflows.dmc.trapped import (
    DMCRunControls,
    controls_to_dict,
    make_grid,
    parse_case,
    parse_seeds,
)

DEFAULT_CASES = "N10_A0,N10_A0.1,N10_A1,N10_A10,N20_A0,N20_A0.1,N20_A1,N20_A10"
DEFAULT_SEEDS = "7001,7002,7003,7004,7005"
DEFAULT_OUTPUT_ROOT = Path("results/dmc/final_matrix/thesis_5seed")

DEFAULT_DT = 0.0025
DEFAULT_WALKERS = 256
DEFAULT_INITIALIZATION_MODE = "lda-rms-logspread"
DEFAULT_INIT_WIDTH_LOG_SIGMA = 0.10
DEFAULT_BREATHING_PREBURN_STEPS = 1000
DEFAULT_BREATHING_PREBURN_LOG_STEP = 0.04
LOCAL_STEP_METHOD = "metropolis"

# The finite-rod N=2 anchor requires a forward-walking projection well beyond
# the earlier seven-unit ladder. The longest lag is 50 oscillator-time units,
# while production retains support through tau_prod=480.
FW_LAG_TIMES = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
# Density needs enough independent transported snapshots to resolve finite-N
# shell structure. Its shorter ladder keeps large density histograms bounded.
DENSITY_FW_LAG_TIMES = (0.0, 2.0, 4.0, 7.0)
STORE_INTERVAL_TAU = 0.025
FW_COLLECTION_INTERVAL_TAU = 0.05
DENSITY_FW_COLLECTION_INTERVAL_TAU = 0.1
LARGE_GRID_DENSITY_FW_COLLECTION_INTERVAL_TAU = 0.5


@dataclass(frozen=True)
class FinalMatrixConfig:
    cases: str = DEFAULT_CASES
    seeds: str = DEFAULT_SEEDS
    burn_tau: float = 60.0
    production_tau: float = 480.0
    grid_extent: float = 35.0
    excluded_volume_margin: float = 35.0
    n_bins: int = 840
    max_density_bin_width: float = 0.20
    ess_resample_fraction: float = 0.35
    pure_fw_block_size_steps: int = 1
    pure_fw_min_block_count: int = 20
    pure_fw_min_walker_weight_ess: float = 30.0
    pure_fw_min_source_ancestor_ess: float = 50.0
    pure_fw_max_source_family_fraction: float = 0.10
    pure_fw_density_plateau_window_lag_count: int = 3
    parallel_workers: int = 5
    plot_formats: str = "png,pdf"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    dry_run: bool = False
    force: bool = False
    continue_on_error: bool = False
    progress: bool = False


@dataclass(frozen=True)
class FinalMatrixResult:
    output_root: Path
    manifest_path: Path | None
    rows: list[dict[str, Any]]
    dry_run: bool

    @property
    def status_counts(self) -> dict[str, int]:
        statuses = sorted({str(record.get("status")) for record in self.rows})
        return {
            status: sum(record.get("status") == status for record in self.rows)
            for status in statuses
        }

    @property
    def status(self) -> str:
        if any(status.startswith("failed") for status in self.status_counts):
            return "failed"
        return "planned" if self.dry_run else "completed"

    @property
    def summary(self) -> dict[str, Any]:
        return {"row_count": len(self.rows), "status_counts": self.status_counts}

    @property
    def artifacts(self) -> dict[str, str | None]:
        return {
            "manifest": str(self.manifest_path) if self.manifest_path is not None else None,
            "output_root": str(self.output_root),
        }

    @property
    def verbose_payload(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path) if self.manifest_path is not None else None,
            "output_root": str(self.output_root),
            "rows": self.rows,
        }


@dataclass(frozen=True)
class RowMethod:
    dt: float
    walkers: int
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


@dataclass(frozen=True)
class _PlannedRow:
    case_id: str
    output_dir: Path
    method: RowMethod
    grid_plan: dict[str, float | int]
    verified_complete: bool
    completion_errors: list[str]
    has_existing_artifacts: bool


def run_final_matrix(config: FinalMatrixConfig, *, repo_root: Path) -> FinalMatrixResult:
    """Plan or dispatch the fixed thesis DMC matrix with verified row reuse."""

    _validate_config(config)
    cases = _parse_cases(config.cases)
    seeds = parse_seeds(config.seeds)
    output_root = config.output_root.expanduser().resolve()
    implementation = implementation_identity(repo_root)
    implementation_fingerprint = _implementation_fingerprint(implementation)

    plans: list[_PlannedRow] = []
    for case_id in cases:
        case_output_dir = output_root / case_id
        method = _row_method(case_id)
        grid_plan = _case_grid_plan(config, case_id, method)
        completed, completion_errors = _verified_completed_row(
            config,
            case_id,
            case_output_dir,
            grid_plan,
            method,
            implementation_fingerprint=implementation_fingerprint,
        )
        has_existing_artifacts = _has_existing_artifacts(case_output_dir)
        if has_existing_artifacts and not completed and not config.force:
            details = "; ".join(completion_errors) or "row is not verified complete"
            raise FileExistsError(
                f"refusing to overwrite existing row {case_id} in {case_output_dir}: "
                f"{details}. Inspect the row or rerun with --force."
            )
        plans.append(
            _PlannedRow(
                case_id=case_id,
                output_dir=case_output_dir,
                method=method,
                grid_plan=grid_plan,
                verified_complete=completed,
                completion_errors=completion_errors,
                has_existing_artifacts=has_existing_artifacts,
            )
        )

    records: list[dict[str, Any]] = []
    for plan in plans:
        case_id = plan.case_id
        case_output_dir = plan.output_dir
        method = plan.method
        grid_plan = plan.grid_plan
        record: dict[str, Any] = {
            "case": case_id,
            "output_dir": str(case_output_dir),
            "summary": str(case_output_dir / "summary.json"),
            "grid_plan": grid_plan,
            "method": _row_method_metadata(method),
        }
        summary_path = case_output_dir / "summary.json"
        if plan.verified_complete and not config.force:
            record["status"] = "skipped_verified_complete"
            record["grid_plan"] = _existing_grid_plan(summary_path, grid_plan)
            record["run_manifest"] = str(case_output_dir / "run_manifest.json")
            record["rerun_command"] = _benchmark_command(
                config,
                case_id,
                case_output_dir,
                grid_plan,
                method,
            )
            records.append(record)
            continue
        if plan.completion_errors and plan.has_existing_artifacts:
            record["existing_artifact_errors"] = plan.completion_errors

        command = _benchmark_command(config, case_id, case_output_dir, grid_plan, method)
        record["command"] = command
        if config.dry_run:
            record["status"] = "planned"
            records.append(record)
            continue

        ensure_dir(case_output_dir)
        completed_process = subprocess.run(
            command,
            cwd=repo_root,
            env=_subprocess_env(repo_root),
            check=False,
        )
        record["returncode"] = completed_process.returncode
        if completed_process.returncode == 0:
            verified, verification_errors = _verified_completed_row(
                config,
                case_id,
                case_output_dir,
                grid_plan,
                method,
                implementation_fingerprint=implementation_fingerprint,
            )
            record["status"] = "completed_verified" if verified else "failed_verification"
            if verification_errors:
                record["verification_errors"] = verification_errors
        else:
            record["status"] = "failed"
        records.append(record)
        _write_matrix_manifest(
            output_root,
            config,
            cases,
            seeds,
            records,
            implementation=implementation,
        )
        if record["status"].startswith("failed") and not config.continue_on_error:
            raise SystemExit(completed_process.returncode or 1)

    if config.dry_run:
        return FinalMatrixResult(
            output_root=output_root,
            manifest_path=None,
            rows=records,
            dry_run=True,
        )

    manifest_path = _write_matrix_manifest(
        output_root,
        config,
        cases,
        seeds,
        records,
        implementation=implementation,
    )
    return FinalMatrixResult(
        output_root=output_root,
        manifest_path=manifest_path,
        rows=records,
        dry_run=False,
    )


def _validate_config(config: FinalMatrixConfig) -> None:
    if config.burn_tau < 0.0 or config.production_tau <= 0.0:
        raise ValueError("burn_tau must be nonnegative and production_tau must be positive")
    if config.grid_extent <= 0.0 or config.excluded_volume_margin < 0.0:
        raise ValueError("grid_extent must be positive and excluded_volume_margin nonnegative")
    if config.n_bins < 2 or config.max_density_bin_width <= 0.0:
        raise ValueError("n_bins must be at least two and max_density_bin_width positive")
    if config.parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    if not 0.0 <= config.ess_resample_fraction <= 1.0:
        raise ValueError("ess_resample_fraction must lie in [0, 1]")
    if config.pure_fw_block_size_steps <= 0 or config.pure_fw_min_block_count <= 0:
        raise ValueError("forward-walking block size and minimum block count must be positive")
    if config.pure_fw_min_walker_weight_ess <= 0.0:
        raise ValueError("pure_fw_min_walker_weight_ess must be positive")
    if config.pure_fw_min_source_ancestor_ess <= 0.0:
        raise ValueError("pure_fw_min_source_ancestor_ess must be positive")
    if not 0.0 < config.pure_fw_max_source_family_fraction <= 1.0:
        raise ValueError("pure_fw_max_source_family_fraction must lie in (0, 1]")
    if config.pure_fw_density_plateau_window_lag_count <= 0:
        raise ValueError("pure_fw_density_plateau_window_lag_count must be positive")
    if not [value.strip() for value in config.plot_formats.split(",") if value.strip()]:
        raise ValueError("at least one plot format is required")


def _parse_cases(value: str) -> list[str]:
    cases = [item.strip() for item in value.split(",") if item.strip()]
    if not cases:
        raise ValueError("at least one case is required")
    if len(cases) != len(set(cases)):
        raise ValueError("case ids must be unique")
    for case_id in cases:
        parse_case(case_id)
    return cases


def _case_grid_plan(
    config: FinalMatrixConfig,
    case_id: str,
    method: RowMethod,
) -> dict[str, float | int]:
    """Plan a finite-support density grid before dispatching one matrix row."""

    case = parse_case(case_id)
    minimum_extent = 0.5 * case.n_particles * case.rod_length
    requested_extent = max(
        config.grid_extent,
        minimum_extent + config.excluded_volume_margin,
    )
    controls = _dmc_controls(
        config,
        method,
        grid_extent=requested_extent,
        n_bins=config.n_bins,
    )
    grid = make_grid(controls, case)
    # The child process reconstructs this value from text. Reusing that rounded
    # representation here keeps command, manifest expectation and bin planning equal.
    planned_extent = _command_float(float(max(abs(grid[0]), abs(grid[-1]))))
    planned_bins = max(
        config.n_bins,
        math.ceil((2.0 * planned_extent) / config.max_density_bin_width) + 1,
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
    density = payload.get("estimates", {}).get("density", {})
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
    config: FinalMatrixConfig,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> list[str]:
    command = [
        sys.executable,
        "experiments/dmc/local/benchmark_packet.py",
        "--case",
        case_id,
        "--seeds",
        config.seeds,
        "--dt",
        _format_number(method.dt),
        "--walkers",
        str(method.walkers),
        "--local-step-method",
        LOCAL_STEP_METHOD,
        "--burn-tau",
        _format_number(config.burn_tau),
        "--production-tau",
        _format_number(config.production_tau),
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
        "--guide-family",
        "reduced-tg",
        "--ess-resample-fraction",
        _format_number(config.ess_resample_fraction),
        "--pure-fw-lags",
        _format_int_tuple(method.pure_fw_lags),
        "--pure-fw-density-lags",
        _format_int_tuple(method.pure_fw_density_lags),
        "--pure-fw-block-size-steps",
        str(config.pure_fw_block_size_steps),
        "--pure-fw-collection-stride-steps",
        str(method.pure_fw_collection_stride_steps),
        "--pure-fw-density-collection-stride-steps",
        str(method.pure_fw_density_collection_stride_steps),
        "--pure-fw-min-block-count",
        str(config.pure_fw_min_block_count),
        "--pure-fw-min-walker-weight-ess",
        _format_number(config.pure_fw_min_walker_weight_ess),
        "--pure-fw-min-source-ancestor-ess",
        _format_number(config.pure_fw_min_source_ancestor_ess),
        "--pure-fw-max-source-family-fraction",
        _format_number(config.pure_fw_max_source_family_fraction),
        "--pure-fw-density-plateau-window-lag-count",
        str(config.pure_fw_density_plateau_window_lag_count),
        "--parallel-workers",
        str(config.parallel_workers),
        "--plot-formats",
        config.plot_formats,
        "--output-dir",
        str(output_dir),
    ]
    if method.relative_alpha is not None:
        command.extend(("--relative-alpha", _format_number(method.relative_alpha)))
    if config.progress:
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
    config: FinalMatrixConfig,
    cases: list[str],
    seeds: list[int],
    records: list[dict[str, Any]],
    *,
    implementation: dict[str, Any],
) -> Path:
    path = output_root / "final_matrix_manifest.json"
    discovered_records = _discover_completed_rows(
        output_root,
        config,
        implementation_fingerprint=_implementation_fingerprint(implementation),
    )
    merged_records = {record["case"]: record for record in [*discovered_records, *records]}
    write_json(
        path,
        {
            "schema_version": "hardrod_final_matrix_v3",
            "requested_cases": cases,
            "seeds": seeds,
            "implementation": implementation,
            "invocation_settings": {
                "local_step_method": LOCAL_STEP_METHOD,
                "burn_tau": config.burn_tau,
                "production_tau": config.production_tau,
                "grid_extent": config.grid_extent,
                "excluded_volume_margin": config.excluded_volume_margin,
                "n_bins": config.n_bins,
                "max_density_bin_width": config.max_density_bin_width,
                "ess_resample_fraction": config.ess_resample_fraction,
                "pure_fw_block_size_steps": config.pure_fw_block_size_steps,
                "pure_fw_min_block_count": config.pure_fw_min_block_count,
                "pure_fw_min_walker_weight_ess": config.pure_fw_min_walker_weight_ess,
                "pure_fw_min_source_ancestor_ess": (config.pure_fw_min_source_ancestor_ess),
                "pure_fw_max_source_family_fraction": (config.pure_fw_max_source_family_fraction),
                "pure_fw_density_plateau_window_lag_count": (
                    config.pure_fw_density_plateau_window_lag_count
                ),
                "parallel_workers": config.parallel_workers,
                "plot_formats": config.plot_formats,
                "calculation": "metropolis_corrected_drift_diffusion_dmc",
                "row_method_policy": {
                    "a_over_aho_0_and_0p1": ("dt=0.0025, walkers=256, reduced-TG default guide"),
                    "N10_a_over_aho_1": ("dt=0.00125, walkers=256, relative-alpha=1.5"),
                    "N20_a_over_aho_1": ("dt=0.000625, walkers=512, relative-alpha=1.86658"),
                    "a_over_aho_10": ("dt=0.00025, walkers=256, reduced-TG default guide"),
                    "density_fw": (
                        "physical lags=(0,2,4,7); snapshot interval=0.1 "
                        "except 0.5 for the large a/a_ho=10 grids"
                    ),
                },
            },
            "rows": [merged_records[case_id] for case_id in sorted(merged_records)],
        },
    )
    return path


def _discover_completed_rows(
    output_root: Path,
    config: FinalMatrixConfig,
    *,
    implementation_fingerprint: str,
) -> list[dict[str, Any]]:
    if not output_root.exists():
        return []
    records: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("N*_A*/summary.json")):
        case_id = summary_path.parent.name
        try:
            parse_case(case_id)
        except ValueError:
            continue
        method = _row_method(case_id)
        grid_plan = _case_grid_plan(config, case_id, method)
        completed, _errors = _verified_completed_row(
            config,
            case_id,
            summary_path.parent,
            grid_plan,
            method,
            implementation_fingerprint=implementation_fingerprint,
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


def _dmc_controls(
    config: FinalMatrixConfig,
    method: RowMethod,
    *,
    grid_extent: float,
    n_bins: int,
) -> DMCRunControls:
    return DMCRunControls(
        dt=method.dt,
        walkers=method.walkers,
        burn_tau=config.burn_tau,
        production_tau=config.production_tau,
        store_every=method.store_every,
        grid_extent=grid_extent,
        n_bins=n_bins,
        ess_resample_fraction=config.ess_resample_fraction,
        local_step_method=LOCAL_STEP_METHOD,
        relative_alpha=method.relative_alpha,
    )


def _verified_completed_row(
    config: FinalMatrixConfig,
    case_id: str,
    output_dir: Path,
    grid_plan: dict[str, float | int],
    method: RowMethod,
    *,
    implementation_fingerprint: str,
) -> tuple[bool, list[str]]:
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False, ["missing summary.json or run_manifest.json"]
    try:
        verified, errors = verify_run_manifest(manifest_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return False, [f"invalid run manifest: {exc}"]
    if not verified:
        return False, errors
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_name") != "dmc_benchmark_packet":
        return False, ["run manifest has the wrong artifact owner"]
    if manifest.get("result_schema_version") != "dmc_benchmark_packet_v2":
        return False, ["run manifest has the wrong result schema"]
    manifest_implementation = manifest.get("provenance", {}).get("implementation", {})
    if manifest_implementation.get("source_tree_sha256") != implementation_fingerprint:
        return False, ["run manifest implementation fingerprint mismatch"]
    config_payload = manifest.get("config")
    if not isinstance(config_payload, dict):
        return False, ["run manifest has no configuration payload"]
    if manifest.get("config_fingerprint") != config_fingerprint(config_payload):
        return False, ["run manifest configuration fingerprint mismatch"]
    expected = _expected_manifest_fields(config, case_id, grid_plan, method)
    mismatches = [
        key for key, value in expected.items() if _nested_value(config_payload, key) != value
    ]
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
    config: FinalMatrixConfig,
    case_id: str,
    grid_plan: dict[str, float | int],
    method: RowMethod,
) -> dict[str, Any]:
    controls = _dmc_controls(
        config,
        method,
        grid_extent=float(grid_plan["grid_extent"]),
        n_bins=int(grid_plan["n_bins"]),
    )
    return {
        "case": case_id,
        "seeds": parse_seeds(config.seeds),
        "controls": controls_to_dict(controls),
        "parallel_workers": config.parallel_workers,
        "collective_rn": None,
        "initialization_mode": method.initialization_mode,
        "init_width_log_sigma": method.init_width_log_sigma,
        "breathing_preburn_steps": method.breathing_preburn_steps,
        "breathing_preburn_log_step": method.breathing_preburn_log_step,
        "guide_family": "reduced-tg",
        "pure_config.lag_steps": list(method.pure_fw_lags),
        "pure_config.density_lag_steps": list(method.pure_fw_density_lags),
        "pure_config.observables": ["r2", "density"],
        "pure_config.observable_source": "raw_r2",
        "pure_config.block_size_steps": config.pure_fw_block_size_steps,
        "pure_config.collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_config.density_collection_stride_steps": (
            method.pure_fw_density_collection_stride_steps
        ),
        "pure_config.min_block_count": config.pure_fw_min_block_count,
        "pure_config.min_walker_weight_ess": config.pure_fw_min_walker_weight_ess,
        "pure_config.min_source_ancestor_ess": config.pure_fw_min_source_ancestor_ess,
        "pure_config.max_source_family_fraction": (config.pure_fw_max_source_family_fraction),
        "pure_config.plateau_window_lag_count": 4,
        "pure_config.density_plateau_window_lag_count": (
            config.pure_fw_density_plateau_window_lag_count
        ),
        "plot_formats": [value.strip() for value in config.plot_formats.split(",")],
    }


def _nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _has_existing_artifacts(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


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


def _row_method(case_id: str) -> RowMethod:
    case = parse_case(case_id)
    if math.isclose(case.rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.00025
        walkers = DEFAULT_WALKERS
        relative_alpha = None
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
    elif math.isclose(case.rod_length, 1.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.000625 if case.n_particles == 20 else 0.00125
        walkers = 512 if case.n_particles == 20 else DEFAULT_WALKERS
        relative_alpha = 1.5 if case.n_particles == 10 else 1.86658
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
    else:
        dt = DEFAULT_DT
        walkers = DEFAULT_WALKERS
        relative_alpha = None
        initialization_mode = DEFAULT_INITIALIZATION_MODE
        preburn_steps = DEFAULT_BREATHING_PREBURN_STEPS
    return RowMethod(
        dt=dt,
        walkers=walkers,
        relative_alpha=relative_alpha,
        initialization_mode=initialization_mode,
        init_width_log_sigma=DEFAULT_INIT_WIDTH_LOG_SIGMA,
        breathing_preburn_steps=preburn_steps,
        breathing_preburn_log_step=DEFAULT_BREATHING_PREBURN_LOG_STEP,
        store_every=_steps_for_tau(STORE_INTERVAL_TAU, dt),
        pure_fw_lags=tuple(_steps_for_tau(tau, dt) for tau in FW_LAG_TIMES),
        pure_fw_density_lags=tuple(_steps_for_tau(tau, dt) for tau in DENSITY_FW_LAG_TIMES),
        pure_fw_collection_stride_steps=_steps_for_tau(FW_COLLECTION_INTERVAL_TAU, dt),
        pure_fw_density_collection_stride_steps=_steps_for_tau(
            (
                LARGE_GRID_DENSITY_FW_COLLECTION_INTERVAL_TAU
                if math.isclose(case.rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12)
                else DENSITY_FW_COLLECTION_INTERVAL_TAU
            ),
            dt,
        ),
    )


def _row_method_metadata(method: RowMethod) -> dict[str, float | int | list[int] | None | str]:
    return {
        "dt": method.dt,
        "walkers": method.walkers,
        "relative_alpha": method.relative_alpha,
        "initialization_mode": method.initialization_mode,
        "store_every": method.store_every,
        "pure_fw_lags": list(method.pure_fw_lags),
        "pure_fw_density_lags": list(method.pure_fw_density_lags),
        "pure_fw_collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_fw_density_collection_stride_steps": (method.pure_fw_density_collection_stride_steps),
    }


def _command_float(value: float) -> float:
    """Return the float reconstructed from the command-line representation."""

    return float(_format_number(value))


def _implementation_fingerprint(implementation: dict[str, Any]) -> str:
    value = implementation.get("source_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("current source tree has no implementation fingerprint")
    return value
