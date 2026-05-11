from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.common.numeric import finite_mean, safe_fraction

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StepTelemetry:
    rn_logk_mean: float = float("nan")
    rn_logq_mean: float = float("nan")
    rn_logw_increment_mean: float = float("nan")
    rn_logw_increment_variance: float = float("nan")


@dataclass(frozen=True)
class AdvanceResult:
    positions: FloatArray
    local_energies: FloatArray
    log_weights: FloatArray
    killed: NDArray[np.bool_]
    telemetry: StepTelemetry


@dataclass
class TraceAccumulator:
    step_count: int = 0
    killed_count: int = 0
    ess_fraction_sum: float = 0.0
    log_weight_span_sum: float = 0.0
    rn_logk_values: list[float] = field(default_factory=list)
    rn_logq_values: list[float] = field(default_factory=list)
    rn_logw_increment_values: list[float] = field(default_factory=list)
    rn_logw_increment_variance_values: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        killed_count: int,
        walker_count: int,
        ess: float,
        log_weight_span: float,
        telemetry: StepTelemetry,
    ) -> None:
        self.step_count += 1
        self.killed_count += int(killed_count)
        self.ess_fraction_sum += safe_fraction(ess, walker_count)
        self.log_weight_span_sum += log_weight_span
        if np.isfinite(telemetry.rn_logk_mean):
            self.rn_logk_values.append(telemetry.rn_logk_mean)
        if np.isfinite(telemetry.rn_logq_mean):
            self.rn_logq_values.append(telemetry.rn_logq_mean)
        if np.isfinite(telemetry.rn_logw_increment_mean):
            self.rn_logw_increment_values.append(telemetry.rn_logw_increment_mean)
        if np.isfinite(telemetry.rn_logw_increment_variance):
            self.rn_logw_increment_variance_values.append(telemetry.rn_logw_increment_variance)

    def to_trace_values(self, *, walker_count: int) -> dict[str, float]:
        if self.step_count <= 0:
            return {
                "ess_fraction": float("nan"),
                "log_weight_span": float("nan"),
                "invalid_proposal_fraction": float("nan"),
                "hard_wall_kill_fraction": float("nan"),
                "rn_logk_mean": float("nan"),
                "rn_logq_mean": float("nan"),
                "rn_logw_increment_mean": float("nan"),
                "rn_logw_increment_variance": float("nan"),
            }
        proposal_count = self.step_count * walker_count
        killed_fraction = safe_fraction(self.killed_count, proposal_count)
        return {
            "ess_fraction": self.ess_fraction_sum / self.step_count,
            "log_weight_span": self.log_weight_span_sum / self.step_count,
            "invalid_proposal_fraction": killed_fraction,
            "hard_wall_kill_fraction": killed_fraction,
            "rn_logk_mean": finite_mean(self.rn_logk_values),
            "rn_logq_mean": finite_mean(self.rn_logq_values),
            "rn_logw_increment_mean": finite_mean(self.rn_logw_increment_values),
            "rn_logw_increment_variance": finite_mean(self.rn_logw_increment_variance_values),
        }
