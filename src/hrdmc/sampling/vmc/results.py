from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
@dataclass(frozen=True)
class VMCAttemptCounts:
    """Exclusive transition outcomes whose sum is the exact attempt count."""
    attempted: int = 0
    accepted: int = 0
    invalid_proposals: int = 0
    nonfinite_proposals: int = 0
    metropolis_rejections: int = 0
    def __post_init__(self) -> None:
        values = (
            self.attempted,
            self.accepted,
            self.invalid_proposals,
            self.nonfinite_proposals,
            self.metropolis_rejections,
        )
        if any(value < 0 for value in values):
            raise ValueError("VMC attempt counts must be non-negative")
        if self.accepted + self.rejected != self.attempted:
            raise ValueError("VMC attempt outcomes must partition attempted transitions")
    @property
    def rejected(self) -> int:
        return self.invalid_proposals + self.nonfinite_proposals + self.metropolis_rejections
    @property
    def acceptance_rate(self) -> float:
        if self.attempted == 0:
            return float("nan")
        return self.accepted / self.attempted
    def plus(self, other: VMCAttemptCounts) -> VMCAttemptCounts:
        return VMCAttemptCounts(
            attempted=self.attempted + other.attempted,
            accepted=self.accepted + other.accepted,
            invalid_proposals=self.invalid_proposals + other.invalid_proposals,
            nonfinite_proposals=self.nonfinite_proposals + other.nonfinite_proposals,
            metropolis_rejections=(self.metropolis_rejections + other.metropolis_rejections),
        )
    def to_dict(self) -> dict[str, int | float]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "invalid_proposals": self.invalid_proposals,
            "nonfinite_proposals": self.nonfinite_proposals,
            "metropolis_rejections": self.metropolis_rejections,
            "acceptance_rate": self.acceptance_rate,
        }
@dataclass(frozen=True)
class VMCTransitionEvent:
    """Post-transition production ensemble emitted to estimator-owned observers."""
    production_step: int
    previous_positions: FloatArray
    positions: FloatArray
    gradients: FloatArray
    local_energies: FloatArray
    valid: BoolArray
    def __post_init__(self) -> None:
        if self.production_step <= 0:
            raise ValueError("production_step must be positive")
        positions = np.asarray(self.positions)
        if positions.ndim != 2:
            raise ValueError("event positions must have shape (walkers, particles)")
        if np.asarray(self.previous_positions).shape != positions.shape:
            raise ValueError("event previous_positions must match event positions")
        walkers = positions.shape[0]
        if np.asarray(self.gradients).shape != positions.shape:
            raise ValueError("event gradients must match event positions")
        for name, values in (("local_energies", self.local_energies), ("valid", self.valid)):
            if np.asarray(values).shape != (walkers,):
                raise ValueError(f"event {name} must have one value per walker")
class VMCObserver(Protocol):
    def record_vmc_transition(self, event: VMCTransitionEvent) -> None:
        """Consume every production transition without retaining histories by default."""
@dataclass(frozen=True)
class MultiplexedVMCObserver:
    """Fan out VMC transition events to independent streaming consumers."""
    observers: tuple[VMCObserver, ...]
    def record_vmc_transition(self, event: VMCTransitionEvent) -> None:
        for observer in self.observers:
            observer.record_vmc_transition(event)
@dataclass(frozen=True)
class VMCStreamingResult:
    """Bounded-memory VMC evolution result with exact phase accounting."""
    burn_in_attempts: VMCAttemptCounts
    production_attempts: VMCAttemptCounts
    production_sample_count: int
    seed: int
    wall_seconds: float
    metadata: dict[str, object]
    def __post_init__(self) -> None:
        if self.production_sample_count <= 0:
            raise ValueError("production_sample_count must be positive")
        if self.wall_seconds < 0.0:
            raise ValueError("wall_seconds must be non-negative")
