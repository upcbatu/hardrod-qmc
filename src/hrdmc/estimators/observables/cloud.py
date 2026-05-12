from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CloudMomentResult:
    mean_square_radius: float
    rms_radius: float
    mean_square_radius_stderr: float


def estimate_cloud_moments(snapshots: FloatArray, *, center: float = 0.0) -> CloudMomentResult:
    """Estimate cloud size moments from sampled open-line coordinates."""
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_particles)")
    if snapshots.shape[0] == 0:
        raise ValueError("at least one snapshot is required")

    per_snapshot = np.mean((snapshots - center) ** 2, axis=1)
    mean_square_radius = float(np.mean(per_snapshot))
    stderr = (
        float(np.std(per_snapshot, ddof=1) / np.sqrt(per_snapshot.size))
        if per_snapshot.size > 1
        else 0.0
    )
    return CloudMomentResult(
        mean_square_radius=mean_square_radius,
        rms_radius=float(np.sqrt(mean_square_radius)),
        mean_square_radius_stderr=stderr,
    )
