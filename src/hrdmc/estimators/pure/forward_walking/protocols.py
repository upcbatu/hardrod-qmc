from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class TransportConventionLike(Protocol):
    @property
    def weight_convention(self) -> str: ...

    @property
    def parent_convention(self) -> str: ...

    @property
    def parent_map_scope(self) -> str: ...


class TransportEventLike(Protocol):
    @property
    def production_step_id(self) -> int | None: ...

    @property
    def positions(self) -> FloatArray: ...

    @property
    def r2_rb_per_walker(self) -> FloatArray | None: ...

    @property
    def log_weights_post_resample(self) -> FloatArray: ...

    @property
    def parent_indices(self) -> IntArray: ...

    @property
    def convention(self) -> TransportConventionLike: ...
