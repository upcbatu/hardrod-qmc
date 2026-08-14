from __future__ import annotations

from dataclasses import dataclass

from hrdmc.estimators.forward_walking.results import PureWalkingResult
from hrdmc.sampling.dmc.results import DMCStreamingSummary


@dataclass(frozen=True)
class TrappedTGAnchor:
    n_particles: int
    omega: float
    @property
    def anchor_id(self) -> str:
        return f"trapped_tg_N{self.n_particles}_A0"
@dataclass(frozen=True)
class TrappedTGSeedRun:
    seed: int
    dmc_summary: DMCStreamingSummary
    pure_result: PureWalkingResult
