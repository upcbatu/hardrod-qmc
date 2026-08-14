from __future__ import annotations

import numpy as np

from hrdmc.statistics.blocking import blocking_standard_error
from hrdmc.statistics.streaming import RunningHistogram, RunningStats


def test_running_statistics_match_direct_moments_and_merge() -> None:
    values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    direct = RunningStats.empty().update_many(values)
    merged = (
        RunningStats.empty()
        .update_many(values[:2])
        .merge(RunningStats.empty().update_many(values[2:]))
    )

    assert direct.count == values.size
    np.testing.assert_allclose(direct.mean, np.mean(values))
    np.testing.assert_allclose(direct.variance, np.var(values, ddof=1))
    np.testing.assert_allclose(merged.mean, direct.mean)
    np.testing.assert_allclose(merged.variance, direct.variance)


def test_running_histogram_preserves_weight_under_merge() -> None:
    grid = np.asarray([-1.0, 0.0, 1.0])
    values = np.asarray([-1.0, 0.0, 1.0, 3.0])
    weights = np.asarray([0.5, 1.0, 1.5, 2.0])
    direct = RunningHistogram.from_centers(grid).update(values, weights)
    merged = (
        RunningHistogram.from_centers(grid)
        .update(values[:2], weights[:2])
        .merge(RunningHistogram.from_centers(grid).update(values[2:], weights[2:]))
    )

    np.testing.assert_allclose(direct.counts, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(merged.counts, direct.counts)
    assert direct.density_integral == 3.0
    assert direct.lost_sample_count == 1
    assert direct.lost_weight == 2.0


def test_blocking_curve_uses_successive_power_of_two_blocks() -> None:
    result = blocking_standard_error(np.arange(64, dtype=float), min_blocks=8)

    np.testing.assert_array_equal(result.block_sizes, [1.0, 2.0, 4.0, 8.0])
    np.testing.assert_array_equal(result.n_blocks, [64.0, 32.0, 16.0, 8.0])
    assert np.all(result.stderr > 0.0)
