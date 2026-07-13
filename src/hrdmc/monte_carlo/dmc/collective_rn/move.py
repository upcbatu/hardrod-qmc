from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.monte_carlo.dmc.collective_rn.config import CollectiveRNConfig
from hrdmc.monte_carlo.dmc.collective_rn.proposal import sample_collective_mixture
from hrdmc.monte_carlo.dmc.collective_rn.weights import (
    importance_sampled_rn_log_increment,
    rn_log_increment,
)
from hrdmc.monte_carlo.dmc.common.guide_api import evaluate_guide, guide_log_values
from hrdmc.monte_carlo.dmc.common.numeric import finite_mean, finite_variance
from hrdmc.monte_carlo.dmc.common.population import require_live_weight
from hrdmc.monte_carlo.dmc.local.telemetry import DMCAdvanceResult, DMCStepTelemetry
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.systems.propagators import (
    ProposalTransitionKernel,
    TargetTransitionKernel,
    transition_backend,
)
from hrdmc.wavefunctions.api import DMCGuide


@dataclass(frozen=True)
class CollectiveRNMove:
    """Scheduled collective proposal with exact K/Q log-weight correction."""

    config: CollectiveRNConfig
    system: OpenLineHardRodSystem
    target_kernel: TargetTransitionKernel
    proposal_kernel: ProposalTransitionKernel

    def __post_init__(self) -> None:
        self.config.validate()

    @property
    def name(self) -> str:
        return "collective_rn"

    def validate_timestep(self, dt: float) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isclose(self.config.step_tau, dt, rtol=0.0, atol=1.0e-15):
            raise ValueError(
                "collective RN step_tau must equal the local DMC dt because the "
                "scheduled transition replaces one projector-time step"
            )

    def interval_steps(self, dt: float) -> int:
        self.validate_timestep(dt)
        steps = int(round(self.config.cadence_tau / dt))
        if steps <= 0 or not np.isclose(
            steps * dt,
            self.config.cadence_tau,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("collective RN cadence_tau must be an integer multiple of dt")
        return steps

    def advance(
        self,
        *,
        guide: DMCGuide,
        rng: np.random.Generator,
        positions: np.ndarray,
        local_energies: np.ndarray,
        log_weights: np.ndarray,
    ) -> DMCAdvanceResult:
        old_positions = np.asarray(positions, dtype=float)
        proposal = sample_collective_mixture(
            self.config,
            self.system,
            self.proposal_kernel,
            rng,
            old_positions,
        )
        target_log_density = self.target_kernel.log_density(
            old_positions,
            proposal.x_new,
            self.config.step_tau,
        )
        if self.config.include_guide_ratio:
            increment = importance_sampled_rn_log_increment(
                target_log_density,
                proposal.log_q_forward,
                guide_log_values(guide, old_positions),
                guide_log_values(guide, proposal.x_new),
            )
        else:
            increment = rn_log_increment(target_log_density, proposal.log_q_forward)
        trial_energies, valid = evaluate_guide(guide, proposal.x_new)
        killed = (~valid) | (~np.isfinite(increment))
        next_log_weights = np.full_like(log_weights, -np.inf, dtype=float)
        live = ~killed
        next_log_weights[live] = log_weights[live] + increment[live]
        require_live_weight(next_log_weights)
        return DMCAdvanceResult(
            positions=np.where(killed[:, np.newaxis], old_positions, proposal.x_new),
            local_energies=np.where(killed, local_energies, trial_energies),
            log_weights=next_log_weights,
            killed=killed,
            telemetry=DMCStepTelemetry(
                scheduled_log_target_mean=finite_mean(target_log_density),
                scheduled_log_proposal_mean=finite_mean(proposal.log_q_forward),
                scheduled_log_weight_increment_mean=finite_mean(increment),
                scheduled_log_weight_increment_variance=finite_variance(increment),
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "collective_step_tau": self.config.step_tau,
            "collective_cadence_tau": self.config.cadence_tau,
            "include_guide_ratio": self.config.include_guide_ratio,
            "target_backend": transition_backend(self.target_kernel),
            "proposal_backend": transition_backend(self.proposal_kernel),
        }
