from __future__ import annotations

import numpy as np

from hrdmc.analysis import (
    autocorrelation,
    blocking_curve,
    detect_blocking_plateau,
    integrated_autocorrelation_time,
    linear_slope_statistics,
    trace_stationarity_diagnostics,
)


def test_linear_slope_statistics_detects_trend() -> None:
    t = np.arange(12, dtype=float)
    y = 2.0 * t + 1.0

    result = linear_slope_statistics(t, y)

    np.testing.assert_allclose(result.slope, 2.0)
    assert not result.compatible_with_zero


def test_integrated_autocorrelation_time_counts_independent_samples() -> None:
    rng = np.random.default_rng(123)
    values = rng.normal(size=512)

    result = integrated_autocorrelation_time(values)

    assert result.effective_independent_samples > 100
    assert result.window_lag is not None
    np.testing.assert_allclose(autocorrelation(values)[0], 1.0)


def test_trace_stationarity_diagnostics_accepts_flat_noisy_trace() -> None:
    rng = np.random.default_rng(321)
    t = np.arange(256, dtype=float)
    y = 1.0 + 0.05 * rng.normal(size=t.size)

    result = trace_stationarity_diagnostics(t, y)

    assert result.stationarity_clean
    assert result.trend_clean
    assert result.blocking_clean
    assert result.cumulative_drift_clean
    assert result.slope_z_autocorr_adjusted <= 2.0


def test_trace_stationarity_diagnostics_rejects_drifting_trace() -> None:
    rng = np.random.default_rng(456)
    t = np.arange(256, dtype=float)
    y = 1.0 + 0.01 * t + 0.01 * rng.normal(size=t.size)

    result = trace_stationarity_diagnostics(t, y)

    assert not result.stationarity_clean
    assert not result.trend_clean
    assert result.slope_z_autocorr_adjusted > 2.0


def test_trace_stationarity_reports_blocking_fields() -> None:
    rng = np.random.default_rng(789)
    t = np.arange(512, dtype=float)
    y = np.empty_like(t)
    y[0] = 0.0
    for i in range(1, t.size):
        y[i] = 0.95 * y[i - 1] + rng.normal(scale=0.1)

    result = trace_stationarity_diagnostics(t, y)

    assert np.isfinite(result.first_last_blocking_z)
    assert np.isfinite(result.spread_blocking_z)
    assert np.isfinite(result.blocking_stderr)
    assert np.isfinite(result.first_half_mean)
    assert np.isfinite(result.second_half_mean)
    assert np.isfinite(result.first_second_half_z)
    assert np.isfinite(result.cumulative_mean_drift)
    assert result.block_count == 4
    assert len(result.block_means) == 4
    assert result.spread_veto == (
        result.spread_warning
        and not (
            result.trend_clean
            and result.blocking_clean
            and result.cumulative_drift_clean
        )
    )


def test_blocking_plateau_detection_accepts_stable_tail() -> None:
    rng = np.random.default_rng(9001)
    values = rng.normal(size=4096)

    curve = blocking_curve(values, min_blocks=16)
    plateau = detect_blocking_plateau(
        curve.block_sizes,
        curve.n_blocks,
        curve.stderr,
        min_blocks=16,
        window=3,
        rel_tol=0.35,
    )

    assert plateau.plateau_found
    assert plateau.plateau_stderr > 0.0
    assert plateau.plateau_n_blocks >= 16


def test_blocking_plateau_detection_rejects_unstable_curve() -> None:
    block_sizes = np.asarray([1, 2, 4, 8, 16], dtype=float)
    n_blocks = np.asarray([128, 64, 32, 16, 8], dtype=float)
    stderr = np.asarray([0.1, 0.2, 0.4, 0.9, 1.6], dtype=float)

    plateau = detect_blocking_plateau(
        block_sizes,
        n_blocks,
        stderr,
        min_blocks=16,
        window=3,
        rel_tol=0.10,
    )

    assert not plateau.plateau_found
    assert plateau.reason == "NO_GO_NO_BLOCKING_PLATEAU"
