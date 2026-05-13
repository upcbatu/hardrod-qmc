from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SIGMA_FLOOR_ABS = 1.0e-12
DENSITY_INTEGRAL_TOL = 1.0e-6


@dataclass(frozen=True)
class SigmaDisplay:
    text: str
    status: str


def display_sigma(delta: float, stderr: float) -> SigmaDisplay:
    if not np.isfinite(delta) or not np.isfinite(stderr):
        return SigmaDisplay("sigma undefined", "undefined")
    if stderr < SIGMA_FLOOR_ABS:
        return SigmaDisplay("sigma at numerical floor", "stderr_at_floor")
    sigma = abs(delta) / stderr
    if sigma >= 1000.0:
        return SigmaDisplay(">> 100 sigma", "ok")
    return SigmaDisplay(f"{sigma:.1f} sigma", "ok")


def smart_ylim(value: float, reference: float, stderr: float) -> tuple[float, float]:
    sep = abs(value - reference)
    if np.isfinite(stderr) and stderr > SIGMA_FLOOR_ABS:
        error_span = 4.0 * stderr
    elif sep > 0.0:
        error_span = 0.1 * sep
    else:
        error_span = 0.05 * max(abs(value), 1.0)
    margin = max(sep, error_span)
    margin = max(margin, 0.04 * max(abs(value), abs(reference), 1.0))
    return min(value, reference) - margin, max(value, reference) + margin


def finite_float(value: Any, default: float = float("nan")) -> float:  # noqa: ANN401
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def array_or_none(value: object) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array


def density_integral_status(integral: float, n_particles: int) -> str:
    if not np.isfinite(integral):
        return "density integral unavailable"
    error = abs(integral - float(n_particles))
    if error <= DENSITY_INTEGRAL_TOL:
        return f"integral={integral:.4g}"
    return f"density lost={error:.3g}"


def metric_array(payload: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    values = np.asarray([finite_float(payload.get(key)) for key in keys], dtype=float)
    return values if np.all(np.isfinite(values)) else None
