from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.monte_carlo.dmc.collective_rn import (
    CollectiveRNConfig,
    CollectiveRNMove,
)
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
    OpenHardRodTrapGapHProductTargetKernel,
    OpenHardRodTrapGapHTransformProposalKernel,
    OpenHardRodTrapPrimitiveKernel,
    OpenLineHardRodSystem,
    OpenN2HardRodTrapExactKernel,
)

PROPOSAL_FAMILIES = ("harmonic-mehler", "gap-h-transform")
TARGET_FAMILIES = ("primitive", "gap-h-product", "n2-exact-relative")
DEFAULT_PROPOSAL_FAMILY = "gap-h-transform"
DEFAULT_TARGET_FAMILY = "primitive"
DEFAULT_COMPONENT_LOG_SCALES = (-0.015, -0.010, -0.004, 0.0, 0.004, 0.010, 0.015)
DEFAULT_COMPONENT_PROBABILITIES = (0.03, 0.10, 0.22, 0.30, 0.22, 0.10, 0.03)


@dataclass(frozen=True)
class CollectiveRNControls:
    """Controls for the optional collective reconfiguration move.

    Local importance-sampled DMC does not need this object. Passing it to a
    trapped workflow constructs the proposal and target transition kernels and
    schedules the collective RN move at ``cadence_tau``.
    """

    cadence_tau: float = 0.005
    proposal_family: str = DEFAULT_PROPOSAL_FAMILY
    target_family: str = DEFAULT_TARGET_FAMILY
    component_log_scales: tuple[float, ...] = DEFAULT_COMPONENT_LOG_SCALES
    component_probabilities: tuple[float, ...] = DEFAULT_COMPONENT_PROBABILITIES
    include_guide_ratio: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.cadence_tau) or self.cadence_tau <= 0.0:
            raise ValueError("cadence_tau must be finite and positive")
        if self.proposal_family not in PROPOSAL_FAMILIES:
            raise ValueError(f"unknown collective RN proposal family: {self.proposal_family}")
        if self.target_family not in TARGET_FAMILIES:
            raise ValueError(f"unknown collective RN target family: {self.target_family}")
        if not self.component_log_scales:
            raise ValueError("component_log_scales must not be empty")
        if len(self.component_log_scales) != len(self.component_probabilities):
            raise ValueError("component log scales and probabilities must have equal length")
        scales = np.asarray(self.component_log_scales, dtype=float)
        probabilities = np.asarray(self.component_probabilities, dtype=float)
        if not np.all(np.isfinite(scales)):
            raise ValueError("component_log_scales must be finite")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
            raise ValueError("component_probabilities must be finite and positive")
        if not np.isclose(float(np.sum(probabilities)), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("component_probabilities must sum to one")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "collective_rn_enabled": True,
            "collective_cadence_tau": self.cadence_tau,
            "proposal_family": self.proposal_family,
            "target_family": self.target_family,
            "component_log_scales": list(self.component_log_scales),
            "component_probabilities": list(self.component_probabilities),
            "include_guide_ratio": self.include_guide_ratio,
        }


def build_collective_rn_extension(
    *,
    system: OpenLineHardRodSystem,
    trap: HarmonicTrap,
    controls: CollectiveRNControls,
    dt: float,
) -> CollectiveRNMove:
    """Construct the opt-in collective move and its transition kernels."""

    controls.validate()
    proposal_kernel = (
        HarmonicMehlerKernel(trap=trap)
        if controls.proposal_family == "harmonic-mehler"
        else OpenHardRodTrapGapHTransformProposalKernel(system=system, trap=trap)
    )
    if controls.target_family == "n2-exact-relative":
        target_kernel = OpenN2HardRodTrapExactKernel(system=system, trap=trap)
    elif controls.target_family == "gap-h-product":
        target_kernel = OpenHardRodTrapGapHProductTargetKernel(system=system, trap=trap)
    else:
        target_kernel = OpenHardRodTrapPrimitiveKernel(system=system, trap=trap)
    return CollectiveRNMove(
        config=CollectiveRNConfig(
            step_tau=dt,
            cadence_tau=controls.cadence_tau,
            component_log_scales=controls.component_log_scales,
            component_probabilities=controls.component_probabilities,
            include_guide_ratio=controls.include_guide_ratio,
        ),
        system=system,
        target_kernel=target_kernel,
        proposal_kernel=proposal_kernel,
    )


__all__ = [
    "CollectiveRNControls",
    "build_collective_rn_extension",
]
