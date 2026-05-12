from typing import cast

import numpy as np

from hrdmc.analysis.correlated_error import (
    TRIANGULATED_STATUS,
    triangulated_error_estimate,
)


def test_constant_trace_has_zero_correlated_error():
    result = triangulated_error_estimate(np.ones(128))

    assert result.status == TRIANGULATED_STATUS
    assert result.conservative_stderr == 0.0
    assert result.overlap_pair_count >= 1


def test_ar1_trace_inflates_error_above_naive_stderr():
    rng = np.random.default_rng(123)
    values = np.zeros(2000)
    for index in range(1, values.size):
        values[index] = 0.9 * values[index - 1] + rng.normal()

    result = triangulated_error_estimate(values)
    naive = float(np.std(values, ddof=1) / np.sqrt(values.size))

    assert np.isfinite(result.conservative_stderr)
    assert result.conservative_stderr > naive


def test_result_dict_keeps_all_estimator_rows():
    values = np.linspace(-1.0, 1.0, 64)

    payload = triangulated_error_estimate(values).to_dict()
    conservative_stderr = cast(float, payload["conservative_stderr"])
    estimates = cast(list[object], payload["estimates"])

    assert payload["status"]
    assert conservative_stderr >= 0.0
    assert len(estimates) == 3
