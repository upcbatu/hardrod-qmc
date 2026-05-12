from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.diagnostics import (
    plateau_summary,
    rms_delta_stderr,
    schema_status,
    variance_inflation,
)
from hrdmc.estimators.pure.forward_walking.results import (
    LagValue,
    TransportedLagResult,
)

FloatArray = NDArray[np.float64]


def assemble_observable_result(
    *,
    config: PureWalkingConfig,
    observable: str,
    block_values_by_lag: dict[int, list[FloatArray]],
    block_weight_ess_by_lag: dict[int, list[float]],
    mixed_r2_reference: float | None,
    mixed_rms_radius_reference: float | None,
) -> TransportedLagResult:
    values_by_lag: dict[int, LagValue] = {}
    stderr_by_lag: dict[int, LagValue] = {}
    block_count_by_lag: dict[int, int] = {}
    weight_ess_min_by_lag: dict[int, float] = {}
    weight_ess_mean_by_lag: dict[int, float] = {}
    variance_by_lag: dict[int, float] = {}
    for lag in config.lag_steps:
        values = np.asarray(block_values_by_lag.get(lag, []), dtype=float)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        finite_mask = (
            np.all(np.isfinite(values), axis=1)
            if values.size
            else np.zeros(0, dtype=bool)
        )
        finite = values[finite_mask]
        block_count_by_lag[lag] = int(finite.shape[0])
        ess_values = np.asarray(block_weight_ess_by_lag.get(lag, []), dtype=float)
        finite_ess = ess_values[np.isfinite(ess_values)]
        weight_ess_min_by_lag[lag] = (
            float(np.min(finite_ess)) if finite_ess.size else 0.0
        )
        weight_ess_mean_by_lag[lag] = (
            float(np.mean(finite_ess)) if finite_ess.size else 0.0
        )
        if finite.shape[0] == 0:
            dim = _observable_dimension(config, observable)
            nan_vec = np.full(dim, np.nan, dtype=float)
            values_by_lag[lag] = _as_result_value(nan_vec)
            stderr_by_lag[lag] = _as_result_value(nan_vec)
            variance_by_lag[lag] = float("inf")
            continue
        mean = np.mean(finite, axis=0)
        stderr = _vector_standard_error(finite)
        values_by_lag[lag] = _as_result_value(mean)
        stderr_by_lag[lag] = _as_result_value(stderr)
        variance_by_lag[lag] = variance_inflation(np.mean(finite, axis=1))
    rms_by_lag = None
    rms_stderr_by_lag = None
    if observable == "r2":
        rms_by_lag = {
            lag: _rms_from_result_value(values_by_lag[lag]) for lag in config.lag_steps
        }
        rms_stderr_by_lag = {
            lag: rms_delta_stderr(
                _scalar_result_value(values_by_lag[lag]),
                _scalar_result_value(stderr_by_lag[lag]),
            )
            for lag in config.lag_steps
        }
    schema = schema_status(
        config=config,
        values_by_lag=values_by_lag,
        mixed_r2_reference=mixed_r2_reference if observable == "r2" else None,
        mixed_rms_radius_reference=mixed_rms_radius_reference
        if observable == "r2"
        else None,
    )
    plateau_value, plateau_stderr, bias_bracket, plateau_status = plateau_summary(
        config=config,
        values_by_lag=values_by_lag,
        stderr_by_lag=stderr_by_lag,
        block_count_by_lag=block_count_by_lag,
        weight_ess_min_by_lag=weight_ess_min_by_lag,
    )
    paper_rms = None
    paper_rms_stderr = None
    if observable == "r2":
        paper_rms = _rms_from_result_value(plateau_value)
        paper_rms_stderr = rms_delta_stderr(
            _scalar_result_value(plateau_value),
            _scalar_result_value(plateau_stderr),
        )
    return TransportedLagResult(
        observable=observable,
        lag_steps=config.lag_steps,
        lag_unit=config.lag_unit,
        values_by_lag=values_by_lag,
        stderr_by_lag=stderr_by_lag,
        block_count_by_lag=block_count_by_lag,
        block_weight_ess_min_by_lag=weight_ess_min_by_lag,
        block_weight_ess_mean_by_lag=weight_ess_mean_by_lag,
        variance_inflation_by_lag=variance_by_lag,
        paper_rms_radius_by_lag=rms_by_lag,
        paper_rms_radius_stderr_by_lag=rms_stderr_by_lag,
        plateau_value=plateau_value,
        plateau_stderr=plateau_stderr,
        paper_rms_radius=paper_rms,
        paper_rms_radius_stderr=paper_rms_stderr,
        bias_bracket=bias_bracket,
        plateau_status=plateau_status,
        schema_status=schema,
        metadata=_observable_metadata(config, observable),
    )


def assemble_r2_result(
    *,
    config: PureWalkingConfig,
    block_values_by_lag: dict[int, list[FloatArray]],
    block_weight_ess_by_lag: dict[int, list[float]],
    mixed_r2_reference: float | None,
    mixed_rms_radius_reference: float | None,
) -> TransportedLagResult:
    return assemble_observable_result(
        config=config,
        observable="r2",
        block_values_by_lag=block_values_by_lag,
        block_weight_ess_by_lag=block_weight_ess_by_lag,
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_radius_reference,
    )


def _vector_standard_error(values: FloatArray) -> FloatArray:
    if values.shape[0] < 2:
        return np.full(values.shape[1], np.nan, dtype=float)
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _as_result_value(value: FloatArray) -> LagValue:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (1,):
        return float(arr[0])
    return arr.copy()


def _scalar_result_value(value: LagValue | None) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1:
        return None
    return float(arr[0])


def _rms_from_result_value(value: LagValue | None) -> float:
    scalar = _scalar_result_value(value)
    if scalar is None or not np.isfinite(scalar) or scalar < 0.0:
        return float("nan")
    return float(np.sqrt(scalar))


def _observable_dimension(config: PureWalkingConfig, observable: str) -> int:
    if observable == "r2":
        return 1
    if observable == "density":
        return int(np.asarray(config.density_bin_edges).size - 1)
    if observable == "pair_distance_density":
        return int(np.asarray(config.pair_bin_edges).size - 1)
    if observable == "structure_factor":
        return int(np.asarray(config.structure_k_values).size)
    raise ValueError(f"unsupported observable: {observable}")


def _observable_metadata(config: PureWalkingConfig, observable: str) -> dict[str, object]:
    if observable == "density":
        return {"bin_edges": np.asarray(config.density_bin_edges, dtype=float).tolist()}
    if observable == "pair_distance_density":
        return {"bin_edges": np.asarray(config.pair_bin_edges, dtype=float).tolist()}
    if observable == "structure_factor":
        return {"k_values": np.asarray(config.structure_k_values, dtype=float).tolist()}
    if observable == "r2":
        return {"paper_rms_radius": "sqrt(aggregated_pure_r2)"}
    return {}
