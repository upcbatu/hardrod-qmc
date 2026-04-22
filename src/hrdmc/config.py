from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardRodSystemConfig:
    """Configuration for a 1D hard-rod system on a periodic ring."""

    n_particles: int
    density: float
    rod_length: float

    @property
    def length(self) -> float:
        return self.n_particles / self.density

    def validate(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.density <= 0:
            raise ValueError("density must be positive")
        if self.rod_length < 0:
            raise ValueError("rod_length must be non-negative")
        if self.density * self.rod_length >= 1.0:
            raise ValueError("packing fraction density * rod_length must be < 1")


@dataclass(frozen=True)
class VMCConfig:
    """Configuration for the first Metropolis/VMC smoke tests."""

    n_steps: int = 20_000
    burn_in: int = 2_000
    thinning: int = 10
    step_size: float = 0.5
    seed: int = 1923

    def validate(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.burn_in < 0 or self.burn_in >= self.n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps")
        if self.thinning <= 0:
            raise ValueError("thinning must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    system: HardRodSystemConfig
    vmc: VMCConfig
    output_dir: str

    def validate(self) -> None:
        self.system.validate()
        self.vmc.validate()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ExperimentConfig:
        cfg = cls(
            system=HardRodSystemConfig(**payload["system"]),
            vmc=VMCConfig(**payload["vmc"]),
            output_dir=payload["output_dir"],
        )
        cfg.validate()
        return cfg


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a JSON experiment config file."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return ExperimentConfig.from_mapping(payload)
