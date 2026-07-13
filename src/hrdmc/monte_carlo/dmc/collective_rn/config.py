from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CollectiveRNConfig:
    """Collective proposal and K/Q correction controls."""

    step_tau: float
    cadence_tau: float = 0.005
    component_log_scales: tuple[float, ...] = (-0.05, 0.0, 0.05)
    component_probabilities: tuple[float, ...] = (0.25, 0.50, 0.25)
    include_guide_ratio: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.step_tau) or self.step_tau <= 0.0:
            raise ValueError("step_tau must be finite and positive")
        if not np.isfinite(self.cadence_tau) or self.cadence_tau <= 0.0:
            raise ValueError("cadence_tau must be finite and positive")
        if len(self.component_log_scales) == 0:
            raise ValueError("at least one collective component is required")
        if len(self.component_log_scales) != len(self.component_probabilities):
            raise ValueError("component_log_scales and component_probabilities must match")
        scales = np.asarray(self.component_log_scales, dtype=float)
        probabilities = np.asarray(self.component_probabilities, dtype=float)
        if not np.all(np.isfinite(scales)):
            raise ValueError("component log-scales must be finite")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
            raise ValueError("component probabilities must be finite and positive")
        if abs(float(np.sum(probabilities)) - 1.0) > 1.0e-12:
            raise ValueError("component probabilities must sum to one")
