from __future__ import annotations

import numpy as np

from hrdmc.analysis.blocking import blocking_standard_error
from hrdmc.analysis.metrics import bias, density_l2_error, mean_squared_error


def test_blocking_returns_sequence() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=1024)
    result = blocking_standard_error(x)
    assert len(result.block_sizes) > 0
    assert len(result.stderr) == len(result.block_sizes)


def test_metrics() -> None:
    b = bias(1.2, 1.0)
    mse = mean_squared_error(b, 0.01)
    assert b > 0
    assert mse > 0


def test_density_l2_error() -> None:
    x = np.linspace(0.0, 1.0, 11)
    estimate = np.ones_like(x)
    reference = np.zeros_like(x)
    np.testing.assert_allclose(density_l2_error(x, estimate, reference), 1.0)
