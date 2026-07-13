from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def rn_log_increment(
    target_log_density: FloatArray,
    proposal_log_density: FloatArray,
) -> FloatArray:
    """Return log K_sys - log Q_theta for an RN-corrected proposal."""

    log_k = np.asarray(target_log_density, dtype=float)
    log_q = np.asarray(proposal_log_density, dtype=float)
    if log_k.shape != log_q.shape:
        raise ValueError("target and proposal log densities must have matching shapes")
    return log_k - log_q


def importance_sampled_rn_log_increment(
    target_log_density: FloatArray,
    proposal_log_density: FloatArray,
    guide_log_old: FloatArray,
    guide_log_new: FloatArray,
) -> FloatArray:
    """RN increment for the importance-sampled distribution Psi_T * Psi."""

    return (
        rn_log_increment(target_log_density, proposal_log_density)
        + np.asarray(guide_log_new, dtype=float)
        - np.asarray(guide_log_old, dtype=float)
    )
