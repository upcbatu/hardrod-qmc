from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNBlockDMCConfig:
    """Algorithm-only RN collective-block configuration.

    Physical parameters such as particle count, rod length, trap strength, and
    center belong to system/propagator and guide objects, not this config.
    """

    tau_block: float = 0.01
    component_log_scales: tuple[float, ...] = (-0.05, 0.0, 0.05)
    component_probabilities: tuple[float, ...] = (0.25, 0.50, 0.25)
    rn_cadence_tau: float = 0.005
    ess_resample_fraction: float = 0.35

    def validate(self) -> None:
        if self.tau_block <= 0.0:
            raise ValueError("tau_block must be positive")
        if self.rn_cadence_tau <= 0.0:
            raise ValueError("rn_cadence_tau must be positive")
        if len(self.component_log_scales) == 0:
            raise ValueError("at least one collective component is required")
        if len(self.component_log_scales) != len(self.component_probabilities):
            raise ValueError("component_log_scales and component_probabilities must match")
        scales = np.asarray(self.component_log_scales, dtype=float)
        probs = np.asarray(self.component_probabilities, dtype=float)
        if not np.all(np.isfinite(scales)):
            raise ValueError("component log-scales must be finite")
        if not np.all(np.isfinite(probs)) or np.any(probs <= 0.0):
            raise ValueError("component probabilities must be finite and positive")
        if abs(float(np.sum(probs)) - 1.0) > 1.0e-12:
            raise ValueError("component probabilities must sum to one")
        if not np.isfinite(self.ess_resample_fraction):
            raise ValueError("ess_resample_fraction must be finite")
        if self.ess_resample_fraction < 0.0 or self.ess_resample_fraction > 1.0:
            raise ValueError("ess_resample_fraction must satisfy 0 <= fraction <= 1")
