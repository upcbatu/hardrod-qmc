from __future__ import annotations

from dataclasses import dataclass

from hrdmc.estimators.pure.forward_walking import PureWalkingResult
from hrdmc.monte_carlo.dmc.local import DMCStreamingSummary


@dataclass(frozen=True)
class TrappedTGAnchor:
    n_particles: int
    omega: float

    @property
    def anchor_id(self) -> str:
        return f"trapped_tg_N{self.n_particles}_A0"


@dataclass(frozen=True)
class HomogeneousRingAnchor:
    n_particles: int
    packing_fraction: float

    @property
    def anchor_id(self) -> str:
        return f"hom_ring_N{self.n_particles}_eta{self.packing_fraction:g}"


@dataclass(frozen=True)
class TrappedTGSeedRun:
    seed: int
    dmc_summary: DMCStreamingSummary
    pure_result: PureWalkingResult
