from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VMCResult:
    snapshots: FloatArray
    acceptance_rate: float
    n_attempted: int
    n_accepted: int
    cpu_seconds: float
    seed: int
    metadata: dict[str, float | int | str]


class MetropolisVMC:
    """Single-chain Metropolis sampler for the prototype VMC pipeline."""

    def __init__(
        self,
        system: HardRodSystem,
        trial: HardRodJastrowTrial,
        step_size: float,
        seed: int = 1234,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.system = system
        self.trial = trial
        self.step_size = step_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self, n_steps: int, burn_in: int = 0, thinning: int = 1) -> VMCResult:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps")
        if thinning <= 0:
            raise ValueError("thinning must be positive")

        start = time.perf_counter()
        current = self.system.initial_lattice(jitter=0.1, seed=self.seed)
        current_log = self.trial.log_value(current)
        if not np.isfinite(current_log):
            raise RuntimeError("initial configuration has invalid trial wavefunction")

        snapshots: list[FloatArray] = []
        accepted = 0

        for step in range(n_steps):
            proposal = current.copy()
            idx = int(self.rng.integers(0, self.system.n_particles))
            proposal[idx] = (proposal[idx] + self.rng.uniform(-self.step_size, self.step_size)) % self.system.length

            proposal_log = self.trial.log_value(proposal)
            if np.isfinite(proposal_log):
                log_ratio = 2.0 * (proposal_log - current_log)
                if log_ratio >= 0.0 or self.rng.random() < np.exp(log_ratio):
                    current = proposal
                    current_log = proposal_log
                    accepted += 1

            if step >= burn_in and ((step - burn_in) % thinning == 0):
                snapshots.append(current.copy())

        cpu = time.perf_counter() - start
        return VMCResult(
            snapshots=np.asarray(snapshots, dtype=float),
            acceptance_rate=accepted / n_steps,
            n_attempted=n_steps,
            n_accepted=accepted,
            cpu_seconds=cpu,
            seed=self.seed,
            metadata={
                "sampler": "MetropolisVMC",
                "n_steps": n_steps,
                "burn_in": burn_in,
                "thinning": thinning,
                "step_size": self.step_size,
                "seed": self.seed,
            },
        )
