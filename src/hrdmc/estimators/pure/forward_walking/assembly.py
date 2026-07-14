from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.estimators.pure.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.diagnostics import (
    genealogy_support_status,
    plateau_summary,
    rms_delta_stderr,
    schema_status,
)
from hrdmc.estimators.pure.forward_walking.results import (
    SCHEMA_INVALID,
    LagValue,
    TransportedLagResult,
)

FloatArray = NDArray[np.float64]


def assemble_observable_result_from_stats(
    *,
    config: PureWalkingConfig,
    observable: str,
    block_mean_by_lag: dict[int, FloatArray],
    block_stderr_by_lag: dict[int, FloatArray],
    block_count_by_lag: dict[int, int],
    weight_ess_min_by_lag: dict[int, float],
    weight_ess_mean_by_lag: dict[int, float],
    source_ancestor_ess_min_by_lag: dict[int, float],
    source_ancestor_ess_mean_by_lag: dict[int, float],
    unique_source_ancestor_min_by_lag: dict[int, int],
    max_source_family_fraction_by_lag: dict[int, float],
    variance_by_lag: dict[int, float],
    mixed_observable_reference: LagValue | None,
    mixed_r2_reference: float | None,
    mixed_rms_radius_reference: float | None,
) -> TransportedLagResult:
    """Assemble a lag result from online block statistics."""

    values_by_lag: dict[int, LagValue] = {}
    stderr_by_lag: dict[int, LagValue] = {}
    for lag in config.lag_steps:
        count = block_count_by_lag.get(lag, 0)
        if count <= 0:
            dim = _observable_dimension(config, observable)
            nan_vec = np.full(dim, np.nan, dtype=float)
            values_by_lag[lag] = _as_result_value(nan_vec, scalar=observable == "r2")
            stderr_by_lag[lag] = _as_result_value(nan_vec, scalar=observable == "r2")
        else:
            values_by_lag[lag] = _as_result_value(
                block_mean_by_lag[lag],
                scalar=observable == "r2",
            )
            stderr_by_lag[lag] = _as_result_value(
                block_stderr_by_lag[lag],
                scalar=observable == "r2",
            )

    rms_by_lag = None
    rms_stderr_by_lag = None
    if observable == "r2":
        rms_by_lag = {lag: _rms_from_result_value(values_by_lag[lag]) for lag in config.lag_steps}
        rms_stderr_by_lag = {
            lag: rms_delta_stderr(
                _scalar_result_value(values_by_lag[lag]),
                _scalar_result_value(stderr_by_lag[lag]),
            )
            for lag in config.lag_steps
        }
    schema = schema_status(
        config=config,
        observable=observable,
        values_by_lag=values_by_lag,
        mixed_observable_reference=mixed_observable_reference,
        mixed_r2_reference=mixed_r2_reference if observable == "r2" else None,
        mixed_rms_radius_reference=mixed_rms_radius_reference if observable == "r2" else None,
    )
    density_accounting = _density_accounting_metadata(
        config=config,
        observable=observable,
        values_by_lag=values_by_lag,
    )
    if density_accounting.get("density_accounting_status") == ("density_normalization_mismatch"):
        schema = SCHEMA_INVALID
    (
        plateau_value,
        plateau_stderr,
        bias_bracket,
        plateau_status,
        plateau_diagnostics,
    ) = plateau_summary(
        config=config,
        observable=observable,
        values_by_lag=values_by_lag,
        stderr_by_lag=stderr_by_lag,
        block_count_by_lag=block_count_by_lag,
        weight_ess_min_by_lag=weight_ess_min_by_lag,
    )
    rms_radius = None
    rms_radius_stderr = None
    if observable == "r2":
        rms_radius = _rms_from_result_value(plateau_value)
        rms_radius_stderr = rms_delta_stderr(
            _scalar_result_value(plateau_value),
            _scalar_result_value(plateau_stderr),
        )
    metadata = _observable_metadata(config, observable)
    metadata.update(
        {
            "min_block_count": config.min_block_count,
            "min_walker_weight_ess": config.min_walker_weight_ess,
            "min_source_ancestor_ess": config.min_source_ancestor_ess,
            "max_source_family_fraction": config.max_source_family_fraction,
            "plateau_sigma_threshold": config.plateau_sigma_threshold,
            "plateau_abs_tolerance": config.plateau_abs_tolerance,
            "plateau_window_lag_count": config.plateau_window_lag_count,
            "density_plateau_relative_l2_tolerance": (config.density_plateau_relative_l2_tolerance),
        }
    )
    metadata.update(
        _schema_reference_metadata(
            config=config,
            observable=observable,
            lag0_value=values_by_lag.get(0),
            mixed_observable_reference=mixed_observable_reference,
        )
    )
    metadata.update(density_accounting)
    genealogy_status, genealogy_diagnostics = genealogy_support_status(
        config=config,
        block_count_by_lag=block_count_by_lag,
        source_ancestor_ess_min_by_lag=source_ancestor_ess_min_by_lag,
        unique_source_ancestor_min_by_lag=unique_source_ancestor_min_by_lag,
        max_source_family_fraction_by_lag=max_source_family_fraction_by_lag,
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
        block_source_ancestor_ess_min_by_lag=source_ancestor_ess_min_by_lag,
        block_source_ancestor_ess_mean_by_lag=source_ancestor_ess_mean_by_lag,
        block_unique_source_ancestor_min_by_lag=unique_source_ancestor_min_by_lag,
        block_max_source_family_fraction_by_lag=max_source_family_fraction_by_lag,
        variance_inflation_by_lag=variance_by_lag,
        rms_radius_by_lag=rms_by_lag,
        rms_radius_stderr_by_lag=rms_stderr_by_lag,
        plateau_value=plateau_value,
        plateau_stderr=plateau_stderr,
        rms_radius=rms_radius,
        rms_radius_stderr=rms_radius_stderr,
        bias_bracket=bias_bracket,
        plateau_status=plateau_status,
        schema_status=schema,
        genealogy_status=genealogy_status,
        genealogy_diagnostics=genealogy_diagnostics,
        plateau_diagnostics=plateau_diagnostics,
        metadata=metadata,
    )


def _as_result_value(value: FloatArray, *, scalar: bool) -> LagValue:
    arr = np.asarray(value, dtype=float)
    if scalar and arr.shape == (1,):
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
        return {
            "bin_edges": np.asarray(config.density_bin_edges, dtype=float).tolist(),
            "density_source": config.density_source,
            "density_com_variance": config.density_com_variance,
            "density_parity_average": config.density_parity_average,
            "density_expected_particles": config.density_expected_particles,
            "density_accounting_abs_tolerance": config.density_accounting_abs_tolerance,
        }
    if observable == "pair_distance_density":
        return {"bin_edges": np.asarray(config.pair_bin_edges, dtype=float).tolist()}
    if observable == "structure_factor":
        return {"k_values": np.asarray(config.structure_k_values, dtype=float).tolist()}
    if observable == "r2":
        return {
            "rms_radius": "sqrt(aggregated_pure_r2)",
            "observable_source": config.observable_source,
            "r2_rb_com_variance": config.r2_rb_com_variance,
        }
    return {}


def _schema_reference_metadata(
    *,
    config: PureWalkingConfig,
    observable: str,
    lag0_value: LagValue | None,
    mixed_observable_reference: LagValue | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "lag0_identity_required": ("lag0_identity" in config.transport_invariant_tests_passed),
        "lag0_reference_available": mixed_observable_reference is not None,
    }
    if lag0_value is None or mixed_observable_reference is None:
        return metadata
    lag0 = np.asarray(lag0_value, dtype=float)
    reference = np.asarray(mixed_observable_reference, dtype=float)
    if lag0.shape != reference.shape:
        metadata["lag0_identity_metric"] = float("inf")
        metadata["lag0_identity_norm"] = "shape_mismatch"
        return metadata
    if observable == "density":
        edges = np.asarray(config.density_bin_edges, dtype=float)
        widths = np.diff(edges)
        if widths.shape == lag0.shape:
            scale = float(np.sqrt(np.sum(reference * reference * widths)))
            if scale > 0.0 and np.isfinite(scale):
                metadata["lag0_identity_metric"] = float(
                    np.sqrt(np.sum((lag0 - reference) ** 2 * widths)) / scale
                )
                metadata["lag0_identity_norm"] = "relative_l2"
                return metadata
    metadata["lag0_identity_metric"] = float(np.max(np.abs(lag0 - reference)))
    metadata["lag0_identity_norm"] = "max_abs"
    return metadata


def _density_accounting_metadata(
    *,
    config: PureWalkingConfig,
    observable: str,
    values_by_lag: dict[int, LagValue],
) -> dict[str, object]:
    if observable != "density" or config.density_expected_particles is None:
        return {"density_accounting_status": "not_evaluated"}
    widths = np.diff(np.asarray(config.density_bin_edges, dtype=float))
    integrals: dict[int, float] = {}
    errors: dict[int, float] = {}
    failed = False
    for lag, value in values_by_lag.items():
        density = np.asarray(value, dtype=float)
        if density.shape != widths.shape or not np.all(np.isfinite(density)):
            continue
        integral = float(np.sum(density * widths))
        error = abs(integral - config.density_expected_particles)
        integrals[lag] = integral
        errors[lag] = error
        failed = failed or (
            not np.isfinite(error) or error > config.density_accounting_abs_tolerance
        )
    status = "density_normalization_mismatch" if failed else "accepted"
    if not integrals:
        status = "not_evaluated"
    return {
        "density_accounting_status": status,
        "density_integral_by_lag": integrals,
        "density_accounting_abs_error_by_lag": errors,
        "density_accounting_abs_error_max": max(errors.values(), default=float("nan")),
    }
