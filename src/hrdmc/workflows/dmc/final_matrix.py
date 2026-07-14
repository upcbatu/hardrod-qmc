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
from hrdmc.workflows.dmc.benchmark_packet.case import BENCHMARK_PACKET_SCHEMA_VERSION
from hrdmc.workflows.dmc.guide_validation import load_validated_contact_guide
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
# Practical thesis reporting resolution for RMS lag equivalence.  This is a
# case-study numerical criterion, not a universal hard-rod or DMC constant;
# Monte Carlo uncertainty and the measured lag bound remain separate outputs.
DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE = 1.0e-3
DEFAULT_PLATEAU_EQUIVALENCE_CONFIDENCE_LEVEL = 0.95
LOCAL_STEP_METHOD = "metropolis"
PURE_FW_R2_SOURCE = "r2_rb"
PURE_FW_DENSITY_SOURCE = "com_rao_blackwell"

# The finite-rod N=2 anchor requires a forward-walking projection well beyond
# the earlier seven-unit ladder. The longest lag is 50 oscillator-time units,
# while production retains support through tau_prod=480.
FW_LAG_TIMES = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
N10_A1_FW_LAG_TIMES = (0.0, 4.0, 6.0, 8.0, 10.0)
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
    max_density_bin_width: float = 0.10
    ess_resample_fraction: float = 0.35
    pure_fw_block_size_steps: int = 1
    pure_fw_min_block_count: int = 20
    pure_fw_min_walker_weight_ess: float = 30.0
    pure_fw_min_source_ancestor_ess: float = 50.0
    pure_fw_max_source_family_fraction: float = 0.10
    pure_fw_rms_plateau_relative_tolerance: float = DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE
    pure_fw_plateau_equivalence_confidence_level: float = (
        DEFAULT_PLATEAU_EQUIVALENCE_CONFIDENCE_LEVEL
    )
    pure_fw_density_plateau_window_lag_count: int = 3
    parallel_workers: int = 5
    plot_formats: str = "png,pdf"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    guide_validation_root: Path | None = None
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
    drift_limiter: str
    guide_family: str
    relative_alpha: float | None
    contact_beta: float | None
    guide_parameter_source: str
    guide_parameter_source_sha256: str | None
    guide_parameter_source_manifest_sha256: str | None
    guide_parameter_source_identity_fingerprint: str | None
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
        method = _row_method(case_id, guide_validation_root=config.guide_validation_root)
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
    if (
        not math.isfinite(config.pure_fw_rms_plateau_relative_tolerance)
        or config.pure_fw_rms_plateau_relative_tolerance < 0.0
    ):
        raise ValueError("pure_fw_rms_plateau_relative_tolerance must be finite and non-negative")
    if not 0.0 < config.pure_fw_plateau_equivalence_confidence_level < 1.0:
        raise ValueError(
            "pure_fw_plateau_equivalence_confidence_level must lie strictly between zero and one"
        )
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
        "--drift-limiter",
        method.drift_limiter,
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
        "--ess-resample-fraction",
        _format_number(config.ess_resample_fraction),
        "--pure-fw-lags",
        _format_int_tuple(method.pure_fw_lags),
        "--pure-fw-density-lags",
        _format_int_tuple(method.pure_fw_density_lags),
        "--pure-fw-observable-source",
        PURE_FW_R2_SOURCE,
        "--pure-fw-density-source",
        PURE_FW_DENSITY_SOURCE,
        "--pure-fw-density-parity-average",
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
        "--pure-fw-rms-plateau-relative-tolerance",
        _format_number(config.pure_fw_rms_plateau_relative_tolerance),
        "--pure-fw-plateau-equivalence-confidence-level",
        _format_number(config.pure_fw_plateau_equivalence_confidence_level),
        "--pure-fw-density-plateau-window-lag-count",
        str(config.pure_fw_density_plateau_window_lag_count),
        "--parallel-workers",
        str(config.parallel_workers),
        "--plot-formats",
        config.plot_formats,
        "--output-dir",
        str(output_dir),
    ]
    if method.guide_parameter_source != "explicit":
        command.extend(("--guide-validation-summary", method.guide_parameter_source))
    else:
        command.extend(("--guide-family", method.guide_family))
    if method.relative_alpha is not None and method.guide_parameter_source == "explicit":
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
            "schema_version": "hardrod_final_matrix_v6",
            "requested_cases": cases,
            "seeds": seeds,
            "implementation": implementation,
            "invocation_settings": {
                "local_step_method": LOCAL_STEP_METHOD,
                "drift_limiter_policy": {
                    "a_over_aho_0": "none",
                    "a_over_aho_0p1": "umrigar",
                    "a_over_aho_1_and_10": "umrigar",
                },
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
                "pure_fw_rms_plateau_relative_tolerance": (
                    config.pure_fw_rms_plateau_relative_tolerance
                ),
                "pure_fw_plateau_equivalence_confidence_level": (
                    config.pure_fw_plateau_equivalence_confidence_level
                ),
                "pure_fw_density_plateau_window_lag_count": (
                    config.pure_fw_density_plateau_window_lag_count
                ),
                "pure_fw_r2_source": PURE_FW_R2_SOURCE,
                "pure_fw_density_source": PURE_FW_DENSITY_SOURCE,
                "pure_fw_density_parity_average": True,
                "parallel_workers": config.parallel_workers,
                "plot_formats": config.plot_formats,
                "calculation": "metropolis_corrected_drift_diffusion_dmc",
                "row_method_policy": {
                    "a_over_aho_0": (
                        "dt=0.0025, walkers=256, reduced-TG default guide, no drift limiter"
                    ),
                    "a_over_aho_0p1": (
                        "dt=0.0025, walkers=256, case-validated optimized guide, Umrigar drift"
                    ),
                    "N10_a_over_aho_1": (
                        "dt=0.00125, walkers=512, case-validated optimized guide, Umrigar drift; "
                        "R2 physical lags=(0,4,6,8,10)"
                    ),
                    "N20_a_over_aho_1": (
                        "dt=0.000625, walkers=512, case-validated optimized guide, Umrigar drift"
                    ),
                    "N10_a_over_aho_10": (
                        "dt=0.000125, walkers=256, case-validated optimized guide, Umrigar drift"
                    ),
                    "N20_a_over_aho_10": (
                        "dt=0.00025, walkers=512, case-validated optimized guide, Umrigar drift"
                    ),
                    "density_fw": (
                        "physical lags=(0,2,4,7); snapshot interval=0.1 "
                        "except 0.5 for the large a/a_ho=10 grids"
                    ),
                    "rms_plateau": (
                        f"{100.0 * config.pure_fw_plateau_equivalence_confidence_level:g}% "
                        "family-wise paired-seed Bonferroni equivalence; relative RMS "
                        f"margin={100.0 * config.pure_fw_rms_plateau_relative_tolerance:g}%"
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
        method = _row_method(case_id, guide_validation_root=config.guide_validation_root)
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
        drift_limiter=method.drift_limiter,
        relative_alpha=method.relative_alpha,
        contact_beta=method.contact_beta,
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
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"invalid benchmark summary: {exc}"]
    if manifest.get("run_name") != "dmc_benchmark_packet":
        return False, ["run manifest has the wrong artifact owner"]
    if manifest.get("result_schema_version") != BENCHMARK_PACKET_SCHEMA_VERSION:
        return False, ["run manifest has the wrong result schema"]
    if manifest.get("status") != "accepted":
        return False, [f"run manifest scientific status is {manifest.get('status')!r}"]
    if summary.get("schema_version") != BENCHMARK_PACKET_SCHEMA_VERSION:
        return False, ["benchmark summary has the wrong result schema"]
    if summary.get("case_id") != case_id:
        return False, ["benchmark summary case does not match the planned row"]
    if summary.get("status") != "accepted":
        return False, [f"benchmark summary scientific status is {summary.get('status')!r}"]
    if summary.get("status") != manifest.get("status"):
        return False, ["benchmark summary and run manifest statuses disagree"]
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
        "guide_family": method.guide_family,
        "guide_parameter_source": method.guide_parameter_source,
        "guide_parameter_source_sha256": method.guide_parameter_source_sha256,
        "guide_parameter_source_manifest_sha256": (method.guide_parameter_source_manifest_sha256),
        "guide_parameter_source_identity_fingerprint": (
            method.guide_parameter_source_identity_fingerprint
        ),
        "pure_config.lag_steps": list(method.pure_fw_lags),
        "pure_config.density_lag_steps": list(method.pure_fw_density_lags),
        "pure_config.observables": ["r2", "density"],
        "pure_config.observable_source": PURE_FW_R2_SOURCE,
        "pure_config.density_source": PURE_FW_DENSITY_SOURCE,
        "pure_config.density_parity_average": True,
        "pure_config.block_size_steps": config.pure_fw_block_size_steps,
        "pure_config.collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_config.density_collection_stride_steps": (
            method.pure_fw_density_collection_stride_steps
        ),
        "pure_config.min_block_count": config.pure_fw_min_block_count,
        "pure_config.min_walker_weight_ess": config.pure_fw_min_walker_weight_ess,
        "pure_config.min_source_ancestor_ess": config.pure_fw_min_source_ancestor_ess,
        "pure_config.max_source_family_fraction": (config.pure_fw_max_source_family_fraction),
        "pure_config.rms_plateau_relative_tolerance": (
            config.pure_fw_rms_plateau_relative_tolerance
        ),
        "pure_config.plateau_equivalence_confidence_level": (
            config.pure_fw_plateau_equivalence_confidence_level
        ),
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


def _row_method(
    case_id: str,
    *,
    guide_validation_root: Path | None,
) -> RowMethod:
    case = parse_case(case_id)
    guide_family = "reduced-tg"
    contact_beta: float | None = None
    guide_parameter_source = "explicit"
    guide_parameter_source_sha256: str | None = None
    guide_parameter_source_manifest_sha256: str | None = None
    guide_parameter_source_identity_fingerprint: str | None = None
    r2_lag_times = FW_LAG_TIMES
    if math.isclose(case.rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.000125 if case.n_particles == 10 else 0.00025
        walkers = 512 if case.n_particles == 20 else DEFAULT_WALKERS
        relative_alpha = None
        drift_limiter = "umrigar"
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
    elif math.isclose(case.rod_length, 1.0, rel_tol=0.0, abs_tol=1e-12):
        dt = 0.000625 if case.n_particles == 20 else 0.00125
        walkers = 512 if case.n_particles == 20 else DEFAULT_WALKERS
        relative_alpha = None
        drift_limiter = "umrigar"
        initialization_mode = "lda-rms-lattice"
        preburn_steps = 0
        if case.n_particles == 10:
            walkers = 512
            r2_lag_times = N10_A1_FW_LAG_TIMES
    else:
        dt = DEFAULT_DT
        walkers = DEFAULT_WALKERS
        relative_alpha = None
        is_finite_rod = case.rod_length > 0.0
        drift_limiter = "umrigar" if is_finite_rod else "none"
        initialization_mode = DEFAULT_INITIALIZATION_MODE
        preburn_steps = DEFAULT_BREATHING_PREBURN_STEPS
    if case.rod_length > 0.0:
        if guide_validation_root is None:
            raise ValueError(
                f"{case.case_id} requires --guide-validation-root with a validated "
                "case-specific optimized guide"
            )
        guide_validation_summary = (
            guide_validation_root.expanduser().resolve()
            / case.case_id
            / "validation"
            / "summary.json"
        )
        validated = load_validated_contact_guide(
            guide_validation_summary,
            case=case,
        )
        guide_family = "contact-corrected-reduced-tg"
        relative_alpha = validated.relative_alpha
        contact_beta = validated.contact_beta
        guide_parameter_source = str(validated.summary_path)
        guide_parameter_source_sha256 = validated.summary_sha256
        guide_parameter_source_manifest_sha256 = validated.manifest_sha256
        guide_parameter_source_identity_fingerprint = validated.identity_fingerprint
    return RowMethod(
        dt=dt,
        walkers=walkers,
        drift_limiter=drift_limiter,
        guide_family=guide_family,
        relative_alpha=relative_alpha,
        contact_beta=contact_beta,
        guide_parameter_source=guide_parameter_source,
        guide_parameter_source_sha256=guide_parameter_source_sha256,
        guide_parameter_source_manifest_sha256=(guide_parameter_source_manifest_sha256),
        guide_parameter_source_identity_fingerprint=(guide_parameter_source_identity_fingerprint),
        initialization_mode=initialization_mode,
        init_width_log_sigma=DEFAULT_INIT_WIDTH_LOG_SIGMA,
        breathing_preburn_steps=preburn_steps,
        breathing_preburn_log_step=DEFAULT_BREATHING_PREBURN_LOG_STEP,
        store_every=_steps_for_tau(STORE_INTERVAL_TAU, dt),
        pure_fw_lags=tuple(_steps_for_tau(tau, dt) for tau in r2_lag_times),
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
        "drift_limiter": method.drift_limiter,
        "guide_family": method.guide_family,
        "relative_alpha": method.relative_alpha,
        "contact_beta": method.contact_beta,
        "guide_parameter_source": method.guide_parameter_source,
        "guide_parameter_source_sha256": method.guide_parameter_source_sha256,
        "guide_parameter_source_manifest_sha256": (method.guide_parameter_source_manifest_sha256),
        "guide_parameter_source_identity_fingerprint": (
            method.guide_parameter_source_identity_fingerprint
        ),
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
