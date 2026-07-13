from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.common.numeric import finite_mean, safe_fraction

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DMCStepTelemetry:
    scheduled_log_target_mean: float = float("nan")
    scheduled_log_proposal_mean: float = float("nan")
    scheduled_log_weight_increment_mean: float = float("nan")
    scheduled_log_weight_increment_variance: float = float("nan")
    local_acceptance_fraction: float = float("nan")
    invalid_proposal_fraction: float = float("nan")
    metropolis_rejection_fraction: float = float("nan")
    drift_norm_max: float = float("nan")
    configuration_esjd: float = float("nan")
    r2_esjd: float = float("nan")
    weighted_free_gap_esjd: float = float("nan")


@dataclass(frozen=True)
class DMCAdvanceResult:
    positions: FloatArray
    local_energies: FloatArray
    log_weights: FloatArray
    killed: NDArray[np.bool_]
    telemetry: DMCStepTelemetry


@dataclass
class TraceAccumulator:
    step_count: int = 0
    killed_count: int = 0
    ess_fraction_sum: float = 0.0
    log_weight_span_sum: float = 0.0
    scheduled_log_target_values: list[float] = field(default_factory=list)
    scheduled_log_proposal_values: list[float] = field(default_factory=list)
    scheduled_log_weight_increment_values: list[float] = field(default_factory=list)
    scheduled_log_weight_increment_variance_values: list[float] = field(default_factory=list)
    local_acceptance_values: list[float] = field(default_factory=list)
    invalid_proposal_values: list[float] = field(default_factory=list)
    metropolis_rejection_values: list[float] = field(default_factory=list)
    drift_norm_max_values: list[float] = field(default_factory=list)
    configuration_esjd_values: list[float] = field(default_factory=list)
    r2_esjd_values: list[float] = field(default_factory=list)
    weighted_free_gap_esjd_values: list[float] = field(default_factory=list)

    def update(
        self,
        *,
        killed_count: int,
        walker_count: int,
        ess: float,
        log_weight_span: float,
        telemetry: DMCStepTelemetry,
    ) -> None:
        self.step_count += 1
        self.killed_count += int(killed_count)
        self.ess_fraction_sum += safe_fraction(ess, walker_count)
        self.log_weight_span_sum += log_weight_span
        if np.isfinite(telemetry.scheduled_log_target_mean):
            self.scheduled_log_target_values.append(telemetry.scheduled_log_target_mean)
        if np.isfinite(telemetry.scheduled_log_proposal_mean):
            self.scheduled_log_proposal_values.append(telemetry.scheduled_log_proposal_mean)
        if np.isfinite(telemetry.scheduled_log_weight_increment_mean):
            self.scheduled_log_weight_increment_values.append(
                telemetry.scheduled_log_weight_increment_mean
            )
        if np.isfinite(telemetry.scheduled_log_weight_increment_variance):
            self.scheduled_log_weight_increment_variance_values.append(
                telemetry.scheduled_log_weight_increment_variance
            )
        if np.isfinite(telemetry.local_acceptance_fraction):
            self.local_acceptance_values.append(telemetry.local_acceptance_fraction)
        if np.isfinite(telemetry.invalid_proposal_fraction):
            self.invalid_proposal_values.append(telemetry.invalid_proposal_fraction)
        if np.isfinite(telemetry.metropolis_rejection_fraction):
            self.metropolis_rejection_values.append(telemetry.metropolis_rejection_fraction)
        if np.isfinite(telemetry.drift_norm_max):
            self.drift_norm_max_values.append(telemetry.drift_norm_max)
        if np.isfinite(telemetry.configuration_esjd):
            self.configuration_esjd_values.append(telemetry.configuration_esjd)
        if np.isfinite(telemetry.r2_esjd):
            self.r2_esjd_values.append(telemetry.r2_esjd)
        if np.isfinite(telemetry.weighted_free_gap_esjd):
            self.weighted_free_gap_esjd_values.append(telemetry.weighted_free_gap_esjd)

    def to_trace_values(self, *, walker_count: int) -> dict[str, float]:
        if self.step_count <= 0:
            return {
                "ess_fraction": float("nan"),
                "log_weight_span": float("nan"),
                "invalid_proposal_fraction": float("nan"),
                "hard_wall_kill_fraction": float("nan"),
                "scheduled_log_target_mean": float("nan"),
                "scheduled_log_proposal_mean": float("nan"),
                "scheduled_log_weight_increment_mean": float("nan"),
                "scheduled_log_weight_increment_variance": float("nan"),
                "local_acceptance_fraction": float("nan"),
                "metropolis_rejection_fraction": float("nan"),
                "drift_norm_max": float("nan"),
                "configuration_esjd": float("nan"),
                "r2_esjd": float("nan"),
                "weighted_free_gap_esjd": float("nan"),
            }
        proposal_count = self.step_count * walker_count
        killed_fraction = safe_fraction(self.killed_count, proposal_count)
        return {
            "ess_fraction": self.ess_fraction_sum / self.step_count,
            "log_weight_span": self.log_weight_span_sum / self.step_count,
            "invalid_proposal_fraction": finite_mean(self.invalid_proposal_values),
            "hard_wall_kill_fraction": killed_fraction,
            "scheduled_log_target_mean": finite_mean(self.scheduled_log_target_values),
            "scheduled_log_proposal_mean": finite_mean(self.scheduled_log_proposal_values),
            "scheduled_log_weight_increment_mean": finite_mean(
                self.scheduled_log_weight_increment_values
            ),
            "scheduled_log_weight_increment_variance": finite_mean(
                self.scheduled_log_weight_increment_variance_values
            ),
            "local_acceptance_fraction": finite_mean(self.local_acceptance_values),
            "metropolis_rejection_fraction": finite_mean(self.metropolis_rejection_values),
            "drift_norm_max": (
                float(np.max(self.drift_norm_max_values))
                if self.drift_norm_max_values
                else float("nan")
            ),
            "configuration_esjd": finite_mean(self.configuration_esjd_values),
            "r2_esjd": finite_mean(self.r2_esjd_values),
            "weighted_free_gap_esjd": finite_mean(self.weighted_free_gap_esjd_values),
        }
