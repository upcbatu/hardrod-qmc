from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.local.telemetry import DMCAdvanceResult
from hrdmc.wavefunctions.api import DMCGuide

FloatArray = NDArray[np.float64]


class ScheduledMove(Protocol):
    """Optional transition inserted at a fixed imaginary-time cadence.

    Implementations own their proposal, correction, and physical dependencies.
    ``advance`` replaces exactly one local DMC step and therefore must represent
    the same projector-time increment ``dt``.  Implementations validate that
    requirement before the run begins.  The local engine owns only scheduling,
    population control, and accounting.
    """

    @property
    def name(self) -> str: ...

    def validate_timestep(self, dt: float) -> None: ...

    def interval_steps(self, dt: float) -> int: ...

    def advance(
        self,
        *,
        guide: DMCGuide,
        rng: np.random.Generator,
        positions: FloatArray,
        local_energies: FloatArray,
        log_weights: FloatArray,
    ) -> DMCAdvanceResult: ...

    def metadata(self) -> dict[str, Any]: ...
