from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DMCConfig:
    """Algorithm controls for local importance-sampled DMC.

    Physical system parameters and optional non-local transition mechanisms
    are injected separately.  This keeps the ordinary DMC engine usable
    without constructing a collective-move configuration.
    """

    ess_resample_fraction: float = 0.35
    local_step_method: str = "metropolis"
    drift_limiter: str = "none"

    def validate(self) -> None:
        if not np.isfinite(self.ess_resample_fraction):
            raise ValueError("ess_resample_fraction must be finite")
        if not 0.0 <= self.ess_resample_fraction <= 1.0:
            raise ValueError("ess_resample_fraction must satisfy 0 <= fraction <= 1")
        if self.local_step_method not in {"euler", "metropolis"}:
            raise ValueError("local_step_method must be 'euler' or 'metropolis'")
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")
        if self.local_step_method != "metropolis" and self.drift_limiter != "none":
            raise ValueError("drift_limiter is only supported for metropolis local steps")
