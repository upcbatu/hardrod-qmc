from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

try:
    from numba import njit
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"


def valid_batch(x: FloatArray, rod_length: float) -> NDArray[np.bool_]:
    x = np.asarray(x, dtype=float)
    if NUMBA_AVAILABLE:
        return _valid_batch_numba(x, float(rod_length))
    return _valid_batch_python(x, float(rod_length))


def reduced_tg_log_batch(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    pair_power: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    x = np.asarray(x, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if NUMBA_AVAILABLE:
        return _reduced_tg_log_batch_numba(
            x,
            offsets,
            float(rod_length),
            float(alpha),
            float(center),
            float(pair_power),
        )
    return _reduced_tg_log_batch_python(
        x,
        offsets,
        rod_length=float(rod_length),
        alpha=float(alpha),
        center=float(center),
        pair_power=float(pair_power),
    )


def reduced_tg_grad_lap_local_batch(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    omega2: float,
    pair_power: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    x = np.asarray(x, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if NUMBA_AVAILABLE:
        return _reduced_tg_grad_lap_local_batch_numba(
            x,
            offsets,
            float(rod_length),
            float(alpha),
            float(center),
            float(omega2),
            float(pair_power),
        )
    return _reduced_tg_grad_lap_local_batch_python(
        x,
        offsets,
        rod_length=float(rod_length),
        alpha=float(alpha),
        center=float(center),
        omega2=float(omega2),
        pair_power=float(pair_power),
    )


def _valid_batch_python(x: FloatArray, rod_length: float) -> NDArray[np.bool_]:
    walkers, n_particles = x.shape
    valid = np.empty(walkers, dtype=bool)
    for walker in range(walkers):
        ok = True
        previous = 0.0
        for particle in range(n_particles):
            value = x[walker, particle]
            if not np.isfinite(value):
                ok = False
                break
            if particle > 0 and value - previous < rod_length:
                ok = False
                break
            previous = value
        valid[walker] = ok
    return valid


def _reduced_tg_log_batch_python(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    pair_power: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    walkers, n_particles = x.shape
    log_values = np.empty(walkers, dtype=float)
    finite = np.empty(walkers, dtype=bool)
    valid = _valid_batch_python(x, rod_length)
    for walker in range(walkers):
        if not valid[walker]:
            log_values[walker] = -np.inf
            finite[walker] = False
            continue
        pair_log = 0.0
        gaussian_sum = 0.0
        row_valid = True
        for i in range(n_particles):
            y_i = x[walker, i] - offsets[i]
            gaussian_sum += (y_i - center) ** 2
            for j in range(i + 1, n_particles):
                gap = (x[walker, j] - offsets[j]) - y_i
                if gap <= 0.0 or not np.isfinite(gap):
                    row_valid = False
                    break
                pair_log += float(np.log(gap))
            if not row_valid:
                break
        if row_valid:
            value = -0.5 * alpha * gaussian_sum + pair_power * pair_log
            log_values[walker] = value
            finite[walker] = bool(np.isfinite(value))
        else:
            log_values[walker] = -np.inf
            finite[walker] = False
    return log_values, finite


def _reduced_tg_grad_lap_local_batch_python(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    omega2: float,
    pair_power: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    walkers, n_particles = x.shape
    grad = np.empty((walkers, n_particles), dtype=float)
    lap = np.empty((walkers, n_particles), dtype=float)
    local = np.empty(walkers, dtype=float)
    finite = np.empty(walkers, dtype=bool)
    valid = _valid_batch_python(x, rod_length)
    for walker in range(walkers):
        trap_sum = 0.0
        local_sum = 0.0
        row_finite = bool(valid[walker])
        for i in range(n_particles):
            y_i = x[walker, i] - offsets[i]
            grad_i = -alpha * (y_i - center)
            lap_i = -alpha
            for j in range(n_particles):
                if i == j:
                    continue
                diff = y_i - (x[walker, j] - offsets[j])
                if diff == 0.0:
                    inv = np.inf
                    inv2 = np.inf
                else:
                    inv = 1.0 / diff
                    inv2 = 1.0 / (diff * diff)
                grad_i += pair_power * inv
                lap_i -= pair_power * inv2
            grad[walker, i] = grad_i
            lap[walker, i] = lap_i
            if not np.isfinite(grad_i) or not np.isfinite(lap_i):
                row_finite = False
            local_sum += lap_i + grad_i * grad_i
            centered = x[walker, i] - center
            trap_sum += centered * centered
        value = -local_sum + 0.5 * omega2 * trap_sum
        local[walker] = value
        finite[walker] = row_finite and np.isfinite(value)
    return grad, lap, local, finite


if NUMBA_AVAILABLE:

    @njit(cache=False, fastmath=False)
    def _valid_batch_numba(x: FloatArray, rod_length: float) -> NDArray[np.bool_]:
        walkers, n_particles = x.shape
        valid = np.empty(walkers, dtype=np.bool_)
        for walker in range(walkers):
            ok = True
            previous = 0.0
            for particle in range(n_particles):
                value = x[walker, particle]
                if not np.isfinite(value):
                    ok = False
                    break
                if particle > 0 and value - previous < rod_length:
                    ok = False
                    break
                previous = value
            valid[walker] = ok
        return valid

    @njit(cache=False, fastmath=False)
    def _reduced_tg_log_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        pair_power: float,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        walkers, n_particles = x.shape
        log_values = np.empty(walkers, dtype=np.float64)
        finite = np.empty(walkers, dtype=np.bool_)
        for walker in range(walkers):
            valid = True
            previous = 0.0
            for particle in range(n_particles):
                value = x[walker, particle]
                if not np.isfinite(value):
                    valid = False
                    break
                if particle > 0 and value - previous < rod_length:
                    valid = False
                    break
                previous = value
            pair_log = 0.0
            gaussian_sum = 0.0
            if valid:
                for i in range(n_particles):
                    y_i = x[walker, i] - offsets[i]
                    gaussian_sum += (y_i - center) * (y_i - center)
                    for j in range(i + 1, n_particles):
                        gap = (x[walker, j] - offsets[j]) - y_i
                        if gap <= 0.0 or not np.isfinite(gap):
                            valid = False
                            break
                        pair_log += np.log(gap)
                    if not valid:
                        break
            if valid:
                value = -0.5 * alpha * gaussian_sum + pair_power * pair_log
                log_values[walker] = value
                finite[walker] = np.isfinite(value)
            else:
                log_values[walker] = -np.inf
                finite[walker] = False
        return log_values, finite

    @njit(cache=False, fastmath=False)
    def _reduced_tg_grad_lap_local_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        omega2: float,
        pair_power: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        walkers, n_particles = x.shape
        grad = np.empty((walkers, n_particles), dtype=np.float64)
        lap = np.empty((walkers, n_particles), dtype=np.float64)
        local = np.empty(walkers, dtype=np.float64)
        finite = np.empty(walkers, dtype=np.bool_)
        for walker in range(walkers):
            valid = True
            previous = 0.0
            for particle in range(n_particles):
                value = x[walker, particle]
                if not np.isfinite(value):
                    valid = False
                    break
                if particle > 0 and value - previous < rod_length:
                    valid = False
                    break
                previous = value
            trap_sum = 0.0
            local_sum = 0.0
            row_finite = valid
            for i in range(n_particles):
                y_i = x[walker, i] - offsets[i]
                grad_i = -alpha * (y_i - center)
                lap_i = -alpha
                for j in range(n_particles):
                    if i == j:
                        continue
                    diff = y_i - (x[walker, j] - offsets[j])
                    if diff == 0.0:
                        inv = np.inf
                        inv2 = np.inf
                    else:
                        inv = 1.0 / diff
                        inv2 = 1.0 / (diff * diff)
                    grad_i += pair_power * inv
                    lap_i -= pair_power * inv2
                grad[walker, i] = grad_i
                lap[walker, i] = lap_i
                if not np.isfinite(grad_i) or not np.isfinite(lap_i):
                    row_finite = False
                local_sum += lap_i + grad_i * grad_i
                centered = x[walker, i] - center
                trap_sum += centered * centered
            value = -local_sum + 0.5 * omega2 * trap_sum
            local[walker] = value
            finite[walker] = row_finite and np.isfinite(value)
        return grad, lap, local, finite

