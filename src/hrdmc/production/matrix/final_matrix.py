from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts.manifest import ensure_dir, write_json
from hrdmc.production.matrix.completion import verified_completed_row as _verified_completed_row
from hrdmc.production.matrix.method import RowMethod, row_method, row_method_metadata
from hrdmc.sampling.dmc.run import parse_seeds
from hrdmc.system.settings import THESIS_CASE_ORDER, DMCRunControls, make_grid, parse_case

DEFAULT_CASES = ",".join(THESIS_CASE_ORDER)
DEFAULT_SEEDS = "7001,7002,7003,7004,7005"
DEFAULT_OUTPUT_ROOT = Path("results/dmc/final_matrix/thesis_5seed")
# Thesis reporting resolution; uncertainty and the measured lag bound remain separate.
DEFAULT_RMS_PLATEAU_RELATIVE_TOLERANCE = 1.0e-3
DEFAULT_PLATEAU_EQUIVALENCE_CONFIDENCE_LEVEL = 0.95
PURE_FW_R2_SOURCE = "r2_rb"
PURE_FW_DENSITY_SOURCE = "com_rao_blackwell"


# N=2 requires the 50-unit lag; production retains support through tau_prod=480.
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
class _PlannedRow:
    case_id: str
    output_dir: Path
    method: RowMethod
    grid_plan: dict[str, float | int]
    verified_complete: bool
    completion_errors: list[str]
    has_existing_artifacts: bool


def run_final_matrix(config: FinalMatrixConfig, *, repo_root: Path) -> FinalMatrixResult:
    _validate_config(config)
    cases = _parse_cases(config.cases)
    seeds = parse_seeds(config.seeds)
    output_root = config.output_root.expanduser().resolve()
    plans = _plan_rows(config, cases, output_root)
    records: list[dict[str, Any]] = []
    for plan in plans:
        record = _run_planned_row(config, repo_root, output_root, cases, seeds, plan)
        records.append(record)
        if not config.dry_run and record["status"] != "skipped_verified_complete":
            _write_matrix_manifest(output_root, config, cases, seeds, records)
        if record["status"].startswith("failed") and not config.continue_on_error:
            raise SystemExit(int(record.get("returncode", 1)) or 1)
    if config.dry_run:
        return FinalMatrixResult(output_root, None, records, True)
    manifest_path = _write_matrix_manifest(output_root, config, cases, seeds, records)
    return FinalMatrixResult(output_root, manifest_path, records, False)


def _plan_rows(
    config: FinalMatrixConfig,
    cases: list[str],
    output_root: Path,
) -> list[_PlannedRow]:
    plans: list[_PlannedRow] = []
    for case_id in cases:
        case_output_dir = output_root / case_id
        method = row_method(case_id, guide_validation_root=config.guide_validation_root)
        grid_plan = _case_grid_plan(config, case_id, method)
        completed, completion_errors = _verified_completed_row(
            config,
            case_id,
            case_output_dir,
            grid_plan,
            method,
        )
        has_existing_artifacts = _has_existing_artifacts(case_output_dir)
        if has_existing_artifacts and not completed and not config.force:
            details = "; ".join(completion_errors) or "case is not verified complete"
            raise FileExistsError(
                f"refusing to overwrite existing case {case_id} in {case_output_dir}: "
                f"{details}. Inspect the case or rerun with --force."
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
    return plans


def _run_planned_row(
    config: FinalMatrixConfig,
    repo_root: Path,
    output_root: Path,
    cases: list[str],
    seeds: list[int],
    plan: _PlannedRow,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case": plan.case_id,
        "output_dir": str(plan.output_dir),
        "summary": str(plan.output_dir / "summary.json"),
        "grid_plan": plan.grid_plan,
        "method": row_method_metadata(plan.method),
    }
    command = _benchmark_command(config, plan.case_id, plan.output_dir, plan.grid_plan, plan.method)
    if plan.verified_complete and not config.force:
        record.update(
            status="skipped_verified_complete",
            grid_plan=_existing_grid_plan(plan.output_dir / "summary.json", plan.grid_plan),
            run_manifest=str(plan.output_dir / "run_manifest.json"),
            rerun_command=command,
        )
        return record
    if plan.completion_errors and plan.has_existing_artifacts:
        record["existing_artifact_errors"] = plan.completion_errors
    record["command"] = command
    if config.dry_run:
        record["status"] = "planned"
        return record
    ensure_dir(plan.output_dir)
    process = subprocess.run(command, cwd=repo_root, env=_subprocess_env(repo_root), check=False)
    record["returncode"] = process.returncode
    if process.returncode:
        record["status"] = "failed"
    else:
        verified, errors = _verified_completed_row(
            config, plan.case_id, plan.output_dir, plan.grid_plan, plan.method
        )
        record["status"] = "completed_verified" if verified else "failed_verification"
        if errors:
            record["verification_errors"] = errors
    return record


def _validate_config(config: FinalMatrixConfig) -> None:
    if config.burn_tau < 0.0 or config.production_tau <= 0.0:
        raise ValueError("burn_tau must be nonnegative and production_tau must be positive")
    if config.grid_extent <= 0.0 or config.excluded_volume_margin < 0.0:
        raise ValueError("grid_extent must be positive and excluded_volume_margin nonnegative")
    if config.n_bins < 2 or config.max_density_bin_width <= 0.0:
        raise ValueError("n_bins must be at least two and max_density_bin_width positive")
    _require_positive(
        parallel_workers=config.parallel_workers,
        pure_fw_min_walker_weight_ess=config.pure_fw_min_walker_weight_ess,
        pure_fw_min_source_ancestor_ess=config.pure_fw_min_source_ancestor_ess,
        pure_fw_density_plateau_window_lag_count=(config.pure_fw_density_plateau_window_lag_count),
    )
    if not 0.0 <= config.ess_resample_fraction <= 1.0:
        raise ValueError("ess_resample_fraction must lie in [0, 1]")
    if config.pure_fw_block_size_steps <= 0 or config.pure_fw_min_block_count <= 0:
        raise ValueError("forward-walking block size and minimum block count must be positive")
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
    if not [value.strip() for value in config.plot_formats.split(",") if value.strip()]:
        raise ValueError("at least one plot format is required")


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


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
    fmt = _format_number
    options = {
        "case": case_id,
        "seeds": config.seeds,
        "dt": fmt(method.dt),
        "walkers": method.walkers,
        "drift-limiter": method.drift_limiter,
        "burn-tau": fmt(config.burn_tau),
        "production-tau": fmt(config.production_tau),
        "store-every": method.store_every,
        "grid-extent": fmt(float(grid_plan["grid_extent"])),
        "n-bins": grid_plan["n_bins"],
        "initialization-mode": method.initialization_mode,
        "init-width-log-sigma": fmt(method.init_width_log_sigma),
        "breathing-preburn-steps": method.breathing_preburn_steps,
        "breathing-preburn-log-step": fmt(method.breathing_preburn_log_step),
        "ess-resample-fraction": fmt(config.ess_resample_fraction),
        "pure-fw-lags": _format_int_tuple(method.pure_fw_lags),
        "pure-fw-density-lags": _format_int_tuple(method.pure_fw_density_lags),
        "pure-fw-observable-source": PURE_FW_R2_SOURCE,
        "pure-fw-density-source": PURE_FW_DENSITY_SOURCE,
        "pure-fw-block-size-steps": config.pure_fw_block_size_steps,
        "pure-fw-collection-stride-steps": method.pure_fw_collection_stride_steps,
        "pure-fw-density-collection-stride-steps": method.pure_fw_density_collection_stride_steps,
        "pure-fw-min-block-count": config.pure_fw_min_block_count,
        "pure-fw-min-walker-weight-ess": fmt(config.pure_fw_min_walker_weight_ess),
        "pure-fw-min-source-ancestor-ess": fmt(config.pure_fw_min_source_ancestor_ess),
        "pure-fw-max-source-family-fraction": fmt(config.pure_fw_max_source_family_fraction),
        "pure-fw-rms-plateau-relative-tolerance": fmt(
            config.pure_fw_rms_plateau_relative_tolerance
        ),
        "pure-fw-plateau-equivalence-confidence-level": fmt(
            config.pure_fw_plateau_equivalence_confidence_level
        ),
        "pure-fw-density-plateau-window-lag-count": config.pure_fw_density_plateau_window_lag_count,
        "parallel-workers": config.parallel_workers,
        "plot-formats": config.plot_formats,
        "output-dir": output_dir,
    }
    command = [sys.executable, "experiments/dmc/local/benchmark_packet.py"]
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    command.append("--pure-fw-density-parity-average")
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
) -> Path:
    path = output_root / "final_matrix_manifest.json"
    discovered_records = _discover_completed_rows(
        output_root,
        config,
    )
    merged_records = {record["case"]: record for record in [*discovered_records, *records]}
    settings = asdict(config)
    settings["output_root"] = str(config.output_root)
    settings["guide_validation_root"] = (
        None if config.guide_validation_root is None else str(config.guide_validation_root)
    )
    settings.update(
        pure_fw_r2_source=PURE_FW_R2_SOURCE,
        pure_fw_density_source=PURE_FW_DENSITY_SOURCE,
        pure_fw_density_parity_average=True,
    )
    write_json(
        path,
        {
            "requested_cases": cases,
            "seeds": seeds,
            "invocation_settings": settings,
            "rows": [merged_records[case_id] for case_id in sorted(merged_records)],
        },
    )
    return path


def _discover_completed_rows(
    output_root: Path,
    config: FinalMatrixConfig,
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
        method = row_method(case_id, guide_validation_root=config.guide_validation_root)
        grid_plan = _case_grid_plan(config, case_id, method)
        completed, _errors = _verified_completed_row(
            config,
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
        drift_limiter=method.drift_limiter,
        relative_alpha=method.relative_alpha,
    )


def _has_existing_artifacts(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def _format_number(value: float) -> str:
    return f"{value:g}"


def _format_int_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _command_float(value: float) -> float:
    return float(_format_number(value))
