from __future__ import annotations

import numpy as np
import pytest

from hrdmc.statistics.blocking import detect_blocking_plateau
from hrdmc.statistics.rank_diagnostics import (
    RANK_DIAGNOSTICS_BETWEEN_CHAIN_DISAGREEMENT,
    RANK_DIAGNOSTICS_OK,
    rank_normalized_diagnostics,
)
from hrdmc.statistics.timeseries import (
    autocorrelation,
    integrated_autocorrelation_time,
    linear_slope_statistics,
)


def test_linear_slope_recovers_an_exact_line() -> None:
    times = np.arange(12, dtype=float)
    result = linear_slope_statistics(times, 2.0 * times + 1.0)

    assert result.slope == pytest.approx(2.0)
    assert not result.compatible_with_zero


def test_alternating_trace_stops_at_first_nonpositive_autocorrelation() -> None:
    values = np.asarray([1.0, -1.0] * 8)
    acf = autocorrelation(values)
    result = integrated_autocorrelation_time(values)

    assert acf[0] == pytest.approx(1.0)
    assert acf[1] < 0.0
    assert result.tau_int_samples == 0.5
    assert result.window_lag == 0
    assert result.effective_independent_samples == values.size


def test_blocking_plateau_selects_the_supported_tail() -> None:
    result = detect_blocking_plateau(
        np.asarray([128, 256, 512, 1024, 2048], dtype=float),
        np.asarray([300, 150, 75, 37, 18], dtype=float),
        np.asarray([0.000342, 0.000361, 0.000329, 0.000327, 0.000390]),
        min_blocks=32,
        window=3,
        rel_tol=0.10,
    )

    assert result.plateau_found
    assert result.plateau_block_size == 1024
    assert result.plateau_n_blocks == 37


def test_blocking_plateau_rejects_a_growing_error_curve() -> None:
    result = detect_blocking_plateau(
        np.asarray([1, 2, 4, 8, 16], dtype=float),
        np.asarray([128, 64, 32, 16, 8], dtype=float),
        np.asarray([0.1, 0.2, 0.4, 0.9, 1.6]),
        min_blocks=16,
        window=3,
        rel_tol=0.10,
    )

    assert not result.plateau_found


def test_rank_diagnostics_match_independent_reference_values() -> None:
    chains = np.asarray(
        [
            [0.10, 0.40, 0.20, 0.80, 0.50, 0.90, 0.70, 1.10, 0.60, 1.00, 0.80, 1.20],
            [0.00, 0.30, 0.10, 0.70, 0.40, 0.80, 0.60, 1.00, 0.50, 0.90, 0.70, 1.10],
            [0.20, 0.50, 0.30, 0.90, 0.60, 1.00, 0.80, 1.20, 0.70, 1.10, 0.90, 1.30],
            [0.15, 0.45, 0.25, 0.85, 0.55, 0.95, 0.75, 1.15, 0.65, 1.05, 0.85, 1.25],
        ]
    )

    result = rank_normalized_diagnostics(chains)

    assert result.status == RANK_DIAGNOSTICS_OK
    assert result.split_rhat == pytest.approx(1.2375392428099208, abs=1.0e-14)
    assert result.bulk_ess == pytest.approx(12.225123965722936, abs=1.0e-13)
    np.testing.assert_allclose(
        result.bulk_ess_per_chain,
        [8.082704029443146, 7.745810746699774, 8.927896510291767, 8.522488167783482],
        rtol=0.0,
        atol=1.0e-13,
    )


def test_rank_diagnostics_detect_between_chain_disagreement() -> None:
    result = rank_normalized_diagnostics(np.vstack((np.zeros(8), np.ones(8), np.full(8, 2.0))))

    assert result.status == RANK_DIAGNOSTICS_BETWEEN_CHAIN_DISAGREEMENT
    assert result.split_rhat == float("inf")
    assert result.bulk_ess == 0.0
