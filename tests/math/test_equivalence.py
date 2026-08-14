from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as student_t

from hrdmc.statistics.equivalence import simultaneous_pairwise_equivalence
from hrdmc.statistics.vector_equivalence import (
    EQUIVALENT,
    unpaired_scalar_equivalence,
    unpaired_vector_equivalence,
)


def test_scalar_equivalence_matches_the_welch_familywise_interval() -> None:
    first = np.asarray([1.00, 1.10, 0.90, 1.05, 0.95])
    second = np.asarray([1.02, 0.98, 1.00, 1.04, 0.96, 1.01])
    result = unpaired_scalar_equivalence(
        first,
        second,
        practical_margin=0.3,
        familywise_confidence=0.95,
        family_size=4,
    )

    first_component = np.var(first, ddof=1) / first.size
    second_component = np.var(second, ddof=1) / second.size
    stderr = np.sqrt(first_component + second_component)
    dof = (first_component + second_component) ** 2 / (
        first_component**2 / (first.size - 1) + second_component**2 / (second.size - 1)
    )
    critical = student_t.ppf(1.0 - 0.05 / 8.0, dof)

    assert result.status == EQUIVALENT
    assert result.standard_error == pytest.approx(stderr)
    assert result.degrees_of_freedom == pytest.approx(dof)
    assert result.critical_value == pytest.approx(critical)
    assert result.absolute_upper_bound == pytest.approx(
        abs(np.mean(first) - np.mean(second)) + critical * stderr
    )


def test_vector_bootstrap_is_seed_order_invariant() -> None:
    first = np.asarray(
        [
            [1.00, 2.00, 1.00],
            [1.02, 1.98, 1.01],
            [0.98, 2.03, 0.99],
            [1.01, 2.01, 1.02],
            [0.99, 1.99, 0.98],
        ]
    )
    second = np.asarray(
        [
            [1.01, 2.01, 1.00],
            [0.99, 1.97, 1.02],
            [1.00, 2.02, 0.98],
            [1.02, 1.99, 1.01],
            [0.98, 2.00, 0.99],
        ]
    )
    kwargs = {
        "feature_weights": np.asarray([0.2, 0.3, 0.5]),
        "scale_profile": np.asarray([1.0, 2.0, 1.0]),
        "practical_margin": 0.1,
        "rng_seed": 20260813,
        "bootstrap_replicates": 2_000,
        "familywise_confidence": 0.95,
        "family_size": 2,
    }

    direct = unpaired_vector_equivalence(first, second, **kwargs)
    permuted = unpaired_vector_equivalence(
        first[[4, 1, 3, 0, 2]], second[[2, 4, 0, 3, 1]], **kwargs
    )

    assert direct.status == EQUIVALENT
    assert permuted == direct


def test_pairwise_equivalence_uses_the_bonferroni_t_bound() -> None:
    values = np.asarray(
        [
            [1.0000, 1.0001, 0.9999],
            [1.0002, 1.0001, 1.0000],
            [0.9998, 0.9999, 1.0000],
            [1.0001, 1.0000, 0.9999],
            [0.9999, 1.0000, 1.0001],
        ]
    )
    result = simultaneous_pairwise_equivalence(
        values, equivalence_margin=5.0e-4, confidence_level=0.95
    )

    assert result.equivalent
    assert result.pair_count == 3
    assert result.critical_value == pytest.approx(3.9607864827701835)
    assert result.simultaneous_upper_bound <= result.equivalence_margin
