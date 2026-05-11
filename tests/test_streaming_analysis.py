from __future__ import annotations

import numpy as np
import pytest

from hrdmc.analysis import RunningHistogram, RunningStats


def test_running_stats_matches_numpy_and_merges() -> None:
    values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    full = RunningStats.empty().update_many(values)
    left = RunningStats.empty().update_many(values[:2])
    right = RunningStats.empty().update_many(values[2:])
    merged = left.merge(right)

    assert full.count == values.size
    np.testing.assert_allclose(full.mean, np.mean(values))
    np.testing.assert_allclose(full.variance, np.var(values, ddof=1))
    np.testing.assert_allclose(merged.mean, full.mean)
    np.testing.assert_allclose(merged.variance, full.variance)


def test_running_histogram_density_integrates_in_grid_weight_and_merges() -> None:
    grid = np.array([-1.0, 0.0, 1.0])
    values = np.array([-1.0, 0.0, 1.0, 3.0])
    weights = np.array([0.5, 1.0, 1.5, 2.0])

    full = RunningHistogram.from_centers(grid).update(values, weights)
    left = RunningHistogram.from_centers(grid).update(values[:2], weights[:2])
    right = RunningHistogram.from_centers(grid).update(values[2:], weights[2:])
    merged = left.merge(right)

    np.testing.assert_allclose(full.counts, np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose(full.density_integral, 3.0)
    assert full.lost_sample_count == 1
    np.testing.assert_allclose(full.lost_weight, 2.0)
    np.testing.assert_allclose(merged.counts, full.counts)
    np.testing.assert_allclose(merged.density_integral, full.density_integral)


def test_running_histogram_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights must match values"):
        RunningHistogram.from_centers(np.array([0.0, 1.0])).update(
            np.array([0.0, 1.0]),
            np.array([1.0]),
        )
