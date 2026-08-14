from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from hrdmc.system.guide_registry import load_validated_reduced_tg_guide
from hrdmc.system.settings import parse_case

DEFAULT_GUIDE_VALIDATION_SUMMARY = (
    Path(__file__).resolve().parents[4] / "data" / "final_matrix_guides" / "summary.json"
)
DEFAULT_DT = 0.0025
DEFAULT_WALKERS = 256
DEFAULT_INITIALIZATION_MODE = "lda-rms-logspread"
DEFAULT_INIT_WIDTH_LOG_SIGMA = 0.10
DEFAULT_BREATHING_PREBURN_STEPS = 1000
DEFAULT_BREATHING_PREBURN_LOG_STEP = 0.04
FW_LAG_TIMES = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
N10_A1_FW_LAG_TIMES = (0.0, 4.0, 6.0, 8.0, 10.0)
DENSITY_FW_LAG_TIMES = (0.0, 2.0, 4.0, 7.0)
STORE_INTERVAL_TAU = 0.025
FW_COLLECTION_INTERVAL_TAU = 0.05
DENSITY_FW_COLLECTION_INTERVAL_TAU = 0.1
LARGE_GRID_DENSITY_FW_COLLECTION_INTERVAL_TAU = 0.5


@dataclass(frozen=True)
class RowMethod:
    dt: float
    walkers: int
    drift_limiter: str
    guide_family: str
    relative_alpha: float | None
    guide_parameter_source: str
    initialization_mode: str
    init_width_log_sigma: float
    breathing_preburn_steps: int
    breathing_preburn_log_step: float
    store_every: int
    pure_fw_lags: tuple[int, ...]
    pure_fw_density_lags: tuple[int, ...]
    pure_fw_collection_stride_steps: int
    pure_fw_density_collection_stride_steps: int


def row_method(case_id: str, *, guide_validation_root: Path | None) -> RowMethod:
    case = parse_case(case_id)
    dt, walkers, initialization_mode, preburn_steps = _base_treatment(
        case.n_particles, case.rod_length
    )
    relative_alpha = None
    source = "explicit"
    if case.rod_length > 0.0:
        summary = DEFAULT_GUIDE_VALIDATION_SUMMARY
        if guide_validation_root is not None:
            summary = (
                guide_validation_root.expanduser().resolve()
                / case.case_id
                / "validation"
                / "summary.json"
            )
        validated = load_validated_reduced_tg_guide(summary, case=case)
        relative_alpha = validated.relative_alpha
        source = str(validated.summary_path)
    r2_lags = N10_A1_FW_LAG_TIMES if case.case_id == "N10_A1" else FW_LAG_TIMES
    return RowMethod(
        dt=dt,
        walkers=walkers,
        drift_limiter="umrigar" if case.rod_length > 0.0 else "none",
        guide_family="reduced-tg",
        relative_alpha=relative_alpha,
        guide_parameter_source=source,
        initialization_mode=initialization_mode,
        init_width_log_sigma=DEFAULT_INIT_WIDTH_LOG_SIGMA,
        breathing_preburn_steps=preburn_steps,
        breathing_preburn_log_step=DEFAULT_BREATHING_PREBURN_LOG_STEP,
        store_every=_steps_for_tau(STORE_INTERVAL_TAU, dt),
        pure_fw_lags=tuple(_steps_for_tau(tau, dt) for tau in r2_lags),
        pure_fw_density_lags=tuple(_steps_for_tau(tau, dt) for tau in DENSITY_FW_LAG_TIMES),
        pure_fw_collection_stride_steps=_steps_for_tau(FW_COLLECTION_INTERVAL_TAU, dt),
        pure_fw_density_collection_stride_steps=_steps_for_tau(
            LARGE_GRID_DENSITY_FW_COLLECTION_INTERVAL_TAU
            if math.isclose(case.rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12)
            else DENSITY_FW_COLLECTION_INTERVAL_TAU,
            dt,
        ),
    )


def row_method_metadata(method: RowMethod) -> dict[str, float | int | list[int] | None | str]:
    return {
        "dt": method.dt,
        "walkers": method.walkers,
        "drift_limiter": method.drift_limiter,
        "guide_family": method.guide_family,
        "relative_alpha": method.relative_alpha,
        "guide_parameter_source": method.guide_parameter_source,
        "initialization_mode": method.initialization_mode,
        "store_every": method.store_every,
        "pure_fw_lags": list(method.pure_fw_lags),
        "pure_fw_density_lags": list(method.pure_fw_density_lags),
        "pure_fw_collection_stride_steps": method.pure_fw_collection_stride_steps,
        "pure_fw_density_collection_stride_steps": method.pure_fw_density_collection_stride_steps,
    }


def _base_treatment(n_particles: int, rod_length: float) -> tuple[float, int, str, int]:
    if math.isclose(rod_length, 10.0, rel_tol=0.0, abs_tol=1e-12):
        return (
            0.000125 if n_particles == 10 else 0.00025,
            512 if n_particles == 20 else DEFAULT_WALKERS,
            "lda-rms-lattice",
            0,
        )
    if math.isclose(rod_length, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return 0.000625 if n_particles == 20 else 0.00125, 512, "lda-rms-lattice", 0
    return DEFAULT_DT, DEFAULT_WALKERS, DEFAULT_INITIALIZATION_MODE, DEFAULT_BREATHING_PREBURN_STEPS


def _steps_for_tau(tau: float, dt: float) -> int:
    if tau == 0.0:
        return 0
    steps = round(tau / dt)
    if steps <= 0 or not math.isclose(steps * dt, tau, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{tau=} is not representable at {dt=}")
    return steps
