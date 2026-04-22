from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
EstimatorFamily = Literal["vmc", "mixed", "extrapolated", "pure"]


@dataclass(frozen=True)
class EstimatorFamilyResult:
    """Observable estimate labeled by estimator family."""

    observable: str
    family: EstimatorFamily
    value: float | FloatArray
    cpu_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def vmc_estimator(
    observable: str,
    value: float | FloatArray,
    cpu_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EstimatorFamilyResult:
    return EstimatorFamilyResult(
        observable=observable,
        family="vmc",
        value=value,
        cpu_seconds=cpu_seconds,
        metadata={} if metadata is None else dict(metadata),
    )


def mixed_estimator(
    observable: str,
    value: float | FloatArray,
    cpu_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EstimatorFamilyResult:
    return EstimatorFamilyResult(
        observable=observable,
        family="mixed",
        value=value,
        cpu_seconds=cpu_seconds,
        metadata={} if metadata is None else dict(metadata),
    )


def pure_estimator(
    observable: str,
    value: float | FloatArray,
    cpu_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EstimatorFamilyResult:
    return EstimatorFamilyResult(
        observable=observable,
        family="pure",
        value=value,
        cpu_seconds=cpu_seconds,
        metadata={} if metadata is None else dict(metadata),
    )


def extrapolated_estimator(
    observable: str,
    mixed: EstimatorFamilyResult,
    vmc: EstimatorFamilyResult,
    cpu_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EstimatorFamilyResult:
    """Build O_ext = 2 O_mixed - O_VMC from compatible estimates."""

    if mixed.observable != observable or vmc.observable != observable:
        raise ValueError("observable labels must match the extrapolated estimate target")
    if mixed.family != "mixed":
        raise ValueError("mixed input must have family='mixed'")
    if vmc.family != "vmc":
        raise ValueError("vmc input must have family='vmc'")

    mixed_value = np.asarray(mixed.value, dtype=float)
    vmc_value = np.asarray(vmc.value, dtype=float)
    if mixed_value.shape != vmc_value.shape:
        raise ValueError("mixed and vmc values must have the same shape")

    value = 2.0 * mixed_value - vmc_value
    if value.ndim == 0:
        result_value: float | FloatArray = float(value)
    else:
        result_value = value

    return EstimatorFamilyResult(
        observable=observable,
        family="extrapolated",
        value=result_value,
        cpu_seconds=cpu_seconds,
        metadata={} if metadata is None else dict(metadata),
    )
