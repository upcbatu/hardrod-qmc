from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics.numba_backend import NUMBA_AVAILABLE, njit

FloatArray = NDArray[np.float64]


def backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"


def valid_batch(x: FloatArray, rod_length: float) -> NDArray[np.bool_]:
    x = np.asarray(x, dtype=float)
    if NUMBA_AVAILABLE:
        return _valid_batch_numba(x, float(rod_length))
    return _valid_batch_python(x, float(rod_length))


def reduced_tg_closed_form_local_energy_batch(
    x: FloatArray,
    *,
    rod_length: float,
    omega: float,
) -> FloatArray:
    """Return the exact local energy of the harmonic reduced-TG guide.

    For ``alpha=omega`` and unit Vandermonde power, the derivative terms of the
    reduced-coordinate guide cancel analytically.  Evaluating the resulting
    free-gap expression avoids subtracting large near-contact terms in the
    generic log-derivative formula.
    """

    x = np.asarray(x, dtype=float)
    walkers, n_particles = x.shape
    if n_particles < 2:
        raise ValueError("reduced-TG guide requires at least two particles")
    free_gaps = np.diff(x, axis=1) - float(rod_length)
    gap_index = np.arange(1, n_particles, dtype=float)
    weights = gap_index * (n_particles - gap_index)
    constant = (
        0.5 * float(omega) * n_particles**2
        + float(omega) ** 2
        * float(rod_length) ** 2
        * n_particles
        * (n_particles**2 - 1)
        / 24.0
    )
    return np.full(walkers, constant, dtype=float) + 0.5 * float(omega) ** 2 * float(
        rod_length
    ) * (free_gaps @ weights)


def reduced_tg_relative_width_local_energy_batch(
    x: FloatArray,
    *,
    rod_length: float,
    omega: float,
    relative_alpha: float,
) -> FloatArray:
    """Return a cancellation-free local energy for the split-width guide.

    The center-of-mass Gaussian retains the harmonic width ``omega`` while the
    reduced internal coordinates use ``relative_alpha``.  This is the standard
    reduced-TG local energy plus the analytic correction from the internal
    Gaussian, so no near-contact ``1/g^2`` cancellation is evaluated.
    """

    x = np.asarray(x, dtype=float)
    walkers, n_particles = x.shape
    if n_particles < 2:
        raise ValueError("reduced-TG guide requires at least two particles")
    base = reduced_tg_closed_form_local_energy_batch(
        x,
        rod_length=rod_length,
        omega=omega,
    )
    delta = float(relative_alpha) - float(omega)
    if delta == 0.0:
        return base
    offsets = float(rod_length) * (
        np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1)
    )
    reduced = x - offsets[np.newaxis, :]
    internal = reduced - np.mean(reduced, axis=1, keepdims=True)
    internal_norm2 = np.sum(internal * internal, axis=1)
    constant = 0.5 * delta * (n_particles * n_particles - 1)
    curvature = float(omega) * delta + 0.5 * delta * delta
    return base + constant - curvature * internal_norm2


def reduced_tg_log_batch(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    relative_alpha: float,
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
            float(relative_alpha),
            float(center),
            float(pair_power),
        )
    return _reduced_tg_log_batch_python(
        x,
        offsets,
        rod_length=float(rod_length),
        alpha=float(alpha),
        relative_alpha=float(relative_alpha),
        center=float(center),
        pair_power=float(pair_power),
    )


def reduced_tg_grad_lap_local_batch(
    x: FloatArray,
    offsets: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    relative_alpha: float,
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
            float(relative_alpha),
            float(center),
            float(omega2),
            float(pair_power),
        )
    return _reduced_tg_grad_lap_local_batch_python(
        x,
        offsets,
        rod_length=float(rod_length),
        alpha=float(alpha),
        relative_alpha=float(relative_alpha),
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
    relative_alpha: float,
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
        reduced = np.empty(n_particles, dtype=float)
        row_valid = True
        for i in range(n_particles):
            y_i = x[walker, i] - offsets[i]
            reduced[i] = y_i - center
            for j in range(i + 1, n_particles):
                gap = (x[walker, j] - offsets[j]) - y_i
                if gap <= 0.0 or not np.isfinite(gap):
                    row_valid = False
                    break
                pair_log += float(np.log(gap))
            if not row_valid:
                break
        if row_valid:
            com = float(np.mean(reduced))
            internal = reduced - com
            value = (
                -0.5 * alpha * n_particles * com * com
                -0.5 * relative_alpha * float(np.sum(internal * internal))
                + pair_power * pair_log
            )
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
    relative_alpha: float,
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
        reduced = x[walker] - offsets - center
        com = float(np.mean(reduced))
        trap_sum = 0.0
        local_sum = 0.0
        row_finite = bool(valid[walker])
        for i in range(n_particles):
            y_i = x[walker, i] - offsets[i]
            internal_i = reduced[i] - com
            grad_i = -alpha * com - relative_alpha * internal_i
            lap_i = -alpha / n_particles - relative_alpha * (1.0 - 1.0 / n_particles)
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
        value = -0.5 * local_sum + 0.5 * omega2 * trap_sum
        local[walker] = value
        finite[walker] = row_finite and np.isfinite(value)
    return grad, lap, local, finite


if NUMBA_AVAILABLE:

    @njit(fastmath=False)
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

    @njit(fastmath=False)
    def _reduced_tg_log_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        rod_length: float,
        alpha: float,
        relative_alpha: float,
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
            reduced_sum = 0.0
            reduced_square_sum = 0.0
            if valid:
                for i in range(n_particles):
                    y_i = x[walker, i] - offsets[i]
                    reduced_i = y_i - center
                    reduced_sum += reduced_i
                    reduced_square_sum += reduced_i * reduced_i
                    for j in range(i + 1, n_particles):
                        gap = (x[walker, j] - offsets[j]) - y_i
                        if gap <= 0.0 or not np.isfinite(gap):
                            valid = False
                            break
                        pair_log += np.log(gap)
                    if not valid:
                        break
            if valid:
                com = reduced_sum / n_particles
                internal_norm2 = reduced_square_sum - n_particles * com * com
                value = (
                    -0.5 * alpha * n_particles * com * com
                    -0.5 * relative_alpha * internal_norm2
                    + pair_power * pair_log
                )
                log_values[walker] = value
                finite[walker] = np.isfinite(value)
            else:
                log_values[walker] = -np.inf
                finite[walker] = False
        return log_values, finite

    @njit(fastmath=False)
    def _reduced_tg_grad_lap_local_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        rod_length: float,
        alpha: float,
        relative_alpha: float,
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
            reduced_sum = 0.0
            if valid:
                for i in range(n_particles):
                    reduced_sum += x[walker, i] - offsets[i] - center
            com = reduced_sum / n_particles
            for i in range(n_particles):
                y_i = x[walker, i] - offsets[i]
                internal_i = y_i - center - com
                grad_i = -alpha * com - relative_alpha * internal_i
                lap_i = -alpha / n_particles - relative_alpha * (1.0 - 1.0 / n_particles)
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
            value = -0.5 * local_sum + 0.5 * omega2 * trap_sum
            local[walker] = value
            finite[walker] = row_finite and np.isfinite(value)
        return grad, lap, local, finite
