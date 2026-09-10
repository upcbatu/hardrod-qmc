from __future__ import annotations

import math


def forward_time_steps(time: float, dt: float) -> int:
    """Convert a nonnegative oscillator time to an exact integer lag."""
    if not math.isfinite(time) or time < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("time must be finite and nonnegative; dt must be finite and positive")
    steps = round(time / dt)
    if not math.isclose(steps * dt, time, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"time {time:g} is not representable at dt={dt:g}; choose an integer lag")
    if time > 0 and steps == 0:
        raise ValueError("positive time must contain at least one step")
    return steps


HO_TRAP_OMEGA = 1.0


def harmonic_oscillator_unit_metadata() -> dict[str, float | str]:
    return {
        "coordinate": "q = x/a_ho",
        "length_unit": "a_ho = sqrt(hbar/(m*Omega))",
        "energy_coordinate": "E_tilde = E/(hbar*Omega)",
        "energy_unit": "hbar*Omega",
        "time_unit": "1/Omega",
        "report_energy_unit": "hbar*Omega",
    }
