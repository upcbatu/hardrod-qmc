from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class DMCGuide(Protocol):
    """Importance-sampling guide required by production DMC engines.

    A guide is stronger than a VMC-only trial amplitude: DMC needs the log
    amplitude, drift derivatives, local energy, and validity semantics.
    """

    def log_value(self, positions: FloatArray) -> float: ...

    def grad_log_value(self, positions: FloatArray) -> FloatArray: ...

    def lap_log_value(self, positions: FloatArray) -> FloatArray: ...

    def local_energy(self, positions: FloatArray) -> float: ...

    def is_valid(self, positions: FloatArray) -> bool: ...


class BatchedDMCGuide(DMCGuide, Protocol):
    """Optional vectorized guide interface used by high-throughput DMC runs."""

    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]: ...

    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]: ...

    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]: ...
