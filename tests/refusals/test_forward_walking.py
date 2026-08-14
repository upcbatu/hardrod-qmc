from __future__ import annotations

import numpy as np
import pytest

from hrdmc.estimators.forward_walking.contributions import weighted_density_profile
from hrdmc.uncertainty.forward_walking.run import _validate_controls


def test_candidate_treatment_must_differ_from_the_anchor() -> None:
    with pytest.raises(ValueError, match="candidate treatment equals the anchor"):
        _validate_controls(
            0.001,
            0.03,
            0.95,
            anchor_treatment=(0.01, 512),
            candidate_treatment=(0.01, 512),
        )


def test_density_estimator_rejects_nonuniform_bins() -> None:
    with pytest.raises(ValueError, match="uniform bin widths"):
        weighted_density_profile(
            np.asarray([[-0.25, 0.25]]),
            bin_edges=np.asarray([-1.0, 0.0, 1.1]),
            walker_weights=np.asarray([1.0]),
            source="com_rao_blackwell",
            com_variance=0.25,
        )
