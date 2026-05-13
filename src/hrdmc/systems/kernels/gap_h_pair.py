from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics.numba_backend import NUMBA_AVAILABLE, njit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def sample_cdf_indices(cdf: FloatArray, rows: IntArray, uniforms: FloatArray) -> IntArray:
    row_flat = np.asarray(rows, dtype=np.int64).reshape(-1)
    u_flat = np.asarray(uniforms, dtype=float).reshape(-1)
    if NUMBA_AVAILABLE:
        out = _sample_cdf_indices_numba(cdf, row_flat, u_flat)
    else:
        out = _sample_cdf_indices_python(cdf, row_flat, u_flat)
    return out.reshape(rows.shape)


def _sample_cdf_indices_python(cdf: FloatArray, rows: IntArray, uniforms: FloatArray) -> IntArray:
    out = np.empty(rows.size, dtype=np.int64)
    for idx, (row, uniform) in enumerate(zip(rows, uniforms, strict=True)):
        out[idx] = np.searchsorted(cdf[row], uniform, side="left")
    return out


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _sample_cdf_indices_numba(
        cdf: FloatArray,
        rows: IntArray,
        uniforms: FloatArray,
    ) -> IntArray:
        out = np.empty(rows.size, dtype=np.int64)
        n_cols = cdf.shape[1]
        for idx in range(rows.size):
            row = rows[idx]
            uniform = uniforms[idx]
            lo = 0
            hi = n_cols
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf[row, mid] < uniform:
                    lo = mid + 1
                else:
                    hi = mid
            out[idx] = lo if lo < n_cols else n_cols - 1
        return out

else:

    def _sample_cdf_indices_numba(
        cdf: FloatArray,
        rows: IntArray,
        uniforms: FloatArray,
    ) -> IntArray:
        return _sample_cdf_indices_python(cdf, rows, uniforms)
