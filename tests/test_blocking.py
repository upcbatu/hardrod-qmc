from __future__ import annotations

import numpy as np

from hrdmc.analysis.blocking import blocking_standard_error
from hrdmc.analysis.metrics import bias, cost_score, mean_squared_error


def test_blocking_returns_sequence() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=1024)
    result = blocking_standard_error(x)
    assert len(result.block_sizes) > 0
    assert len(result.stderr) == len(result.block_sizes)


def test_metrics() -> None:
    b = bias(1.2, 1.0)
    mse = mean_squared_error(b, 0.01)
    score = cost_score(mse, 2.0)
    assert b > 0
    assert mse > 0
    assert score > 0
