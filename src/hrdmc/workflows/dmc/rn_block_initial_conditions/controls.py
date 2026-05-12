from __future__ import annotations

from dataclasses import dataclass

INITIALIZATION_MODES = ("tight-lattice", "lda-rms-lattice", "lda-rms-logspread")


@dataclass(frozen=True)
class RNInitializationControls:
    mode: str = "tight-lattice"
    init_width_log_sigma: float = 0.10
    breathing_preburn_steps: int = 0
    breathing_preburn_log_step: float = 0.04

    def validate(self) -> None:
        if self.mode not in INITIALIZATION_MODES:
            raise ValueError(f"unknown initialization mode: {self.mode}")
        if self.init_width_log_sigma < 0.0:
            raise ValueError("init_width_log_sigma must be non-negative")
        if self.breathing_preburn_steps < 0:
            raise ValueError("breathing_preburn_steps must be non-negative")
        if self.breathing_preburn_log_step < 0.0:
            raise ValueError("breathing_preburn_log_step must be non-negative")
