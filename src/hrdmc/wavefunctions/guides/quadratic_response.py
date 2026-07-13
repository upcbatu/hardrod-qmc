from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import (
    BASE_TRAP_QUADRATIC_COUPLING,
    HarmonicTrap,
    lambda_from_relative_offset,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.api import BatchedDMCGuide

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FixedGuideQuadraticResponse:
    """Evaluate one fixed guide under an auxiliary quadratic Hamiltonian.

    The base oscillator-unit Hamiltonian contains

        0.5 * sum_i (q_i - q0)^2.

    This wrapper leaves the guide amplitude, drift, Laplacian, support, and
    batch implementation unchanged.  It changes only the local energy,

        E_L(lambda) = E_L(0.5) + (lambda - 0.5) * sum_i (q_i - q0)^2.

    Keeping the guide fixed separates the Hamiltonian response from a change
    in variational parameters.  The wrapper requires a batched guide so the
    DMC hot path remains vectorized.
    """

    base_guide: BatchedDMCGuide
    lambda_value: float
    center: float = 0.0
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_value) or self.lambda_value <= 0.0:
            raise ValueError("lambda_value must be positive and finite")
        lambda_from_relative_offset(0.0, lambda0=self.lambda0)
        if not np.isfinite(self.center):
            raise ValueError("center must be finite")
        for method_name in ("batch_log_value", "batch_grad_lap_local", "valid_batch"):
            if not callable(getattr(self.base_guide, method_name, None)):
                raise TypeError("base_guide must implement the batched DMC guide interface")
        for attribute_name in ("system", "trap"):
            if getattr(self.base_guide, attribute_name, None) is None:
                raise TypeError(
                    f"base_guide must expose {attribute_name} for trapped DMC workflows"
                )

    @property
    def delta_lambda(self) -> float:
        return float(self.lambda_value - self.lambda0)

    @property
    def system(self) -> OpenLineHardRodSystem:
        """Delegate hard-rod geometry used by initialization workflows."""

        return cast(OpenLineHardRodSystem, cast(Any, self.base_guide).system)

    @property
    def trap(self) -> HarmonicTrap:
        """Delegate the unperturbed trap metadata from the fixed base guide."""

        return cast(HarmonicTrap, cast(Any, self.base_guide).trap)

    def is_valid(self, positions: FloatArray) -> bool:
        return bool(self.base_guide.is_valid(positions))

    def log_value(self, positions: FloatArray) -> float:
        return float(self.base_guide.log_value(positions))

    def grad_log_value(self, positions: FloatArray) -> FloatArray:
        return np.asarray(self.base_guide.grad_log_value(positions), dtype=float)

    def lap_log_value(self, positions: FloatArray) -> FloatArray:
        return np.asarray(self.base_guide.lap_log_value(positions), dtype=float)

    def local_energy(self, positions: FloatArray) -> float:
        row = np.asarray(positions, dtype=float)
        base_energy = float(self.base_guide.local_energy(row))
        return float(base_energy + self.delta_lambda * np.sum((row - self.center) ** 2))

    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]:
        return np.asarray(self.base_guide.valid_batch(positions), dtype=bool)

    def batch_log_value(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        values, finite = self.base_guide.batch_log_value(positions)
        return np.asarray(values, dtype=float), np.asarray(finite, dtype=bool)

    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        rows = np.asarray(positions, dtype=float)
        grad, lap, local, finite = self.base_guide.batch_grad_lap_local(rows)
        finite_array = np.asarray(finite, dtype=bool)
        response_operator = np.sum((rows - self.center) ** 2, axis=1)
        shifted_local = np.asarray(local, dtype=float) + self.delta_lambda * response_operator
        shifted_local = np.where(finite_array, shifted_local, np.nan)
        return (
            np.asarray(grad, dtype=float),
            np.asarray(lap, dtype=float),
            shifted_local,
            finite_array,
        )

    @property
    def batch_backend(self) -> str:
        backend = getattr(cast(Any, self.base_guide), "batch_backend", "batch")
        return f"fixed_quadratic_response[{backend}]"
