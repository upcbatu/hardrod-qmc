from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics.numba_backend import NUMBA_AVAILABLE, njit

FloatArray = NDArray[np.float64]


def gap_h_guide_backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"


def gap_h_log_batch(
    x: FloatArray,
    offsets: FloatArray,
    y_grid: FloatArray,
    log_correction: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    pair_power: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    if NUMBA_AVAILABLE:
        return _gap_h_log_batch_numba(
            np.asarray(x, dtype=float),
            np.asarray(offsets, dtype=float),
            np.asarray(y_grid, dtype=float),
            np.asarray(log_correction, dtype=float),
            float(rod_length),
            float(alpha),
            float(center),
            float(pair_power),
        )
    return _gap_h_log_batch_python(
        np.asarray(x, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(y_grid, dtype=float),
        np.asarray(log_correction, dtype=float),
        rod_length=float(rod_length),
        alpha=float(alpha),
        center=float(center),
        pair_power=float(pair_power),
    )


def gap_h_grad_lap_local_batch(
    x: FloatArray,
    offsets: FloatArray,
    y_grid: FloatArray,
    grad_correction: FloatArray,
    lap_correction: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    omega2: float,
    pair_power: float,
    n2_total_energy: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    if NUMBA_AVAILABLE:
        return _gap_h_grad_lap_local_batch_numba(
            np.asarray(x, dtype=float),
            np.asarray(offsets, dtype=float),
            np.asarray(y_grid, dtype=float),
            np.asarray(grad_correction, dtype=float),
            np.asarray(lap_correction, dtype=float),
            float(rod_length),
            float(alpha),
            float(center),
            float(omega2),
            float(pair_power),
            float(n2_total_energy),
        )
    return _gap_h_grad_lap_local_batch_python(
        np.asarray(x, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(y_grid, dtype=float),
        np.asarray(grad_correction, dtype=float),
        np.asarray(lap_correction, dtype=float),
        rod_length=float(rod_length),
        alpha=float(alpha),
        center=float(center),
        omega2=float(omega2),
        pair_power=float(pair_power),
        n2_total_energy=float(n2_total_energy),
    )


def _interp_uniform(x: float, grid: FloatArray, values: FloatArray) -> float:
    if x <= grid[0]:
        return float(values[0])
    last = grid.size - 1
    if x >= grid[last]:
        return float(values[last])
    position = (x - grid[0]) / (grid[1] - grid[0])
    left = max(0, min(int(position), last - 1))
    t = position - left
    return float((1.0 - t) * values[left] + t * values[left + 1])


def _gap_h_log_batch_python(
    x: FloatArray,
    offsets: FloatArray,
    y_grid: FloatArray,
    log_correction: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    pair_power: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    walkers, n_particles = x.shape
    values = np.empty(walkers, dtype=float)
    finite = np.empty(walkers, dtype=bool)
    y_max = float(y_grid[-1])
    for walker in range(walkers):
        valid = True
        previous = 0.0
        for particle in range(n_particles):
            value = x[walker, particle]
            if not np.isfinite(value) or (particle > 0 and value - previous <= rod_length):
                valid = False
                break
            previous = value
        pair_log = 0.0
        gaussian_sum = 0.0
        gap_corr = 0.0
        if valid:
            for i in range(n_particles):
                y_i = x[walker, i] - offsets[i]
                gaussian_sum += (y_i - center) ** 2
                if i < n_particles - 1:
                    gap = x[walker, i + 1] - x[walker, i]
                    if gap > y_max:
                        valid = False
                        break
                    gap_corr += _interp_uniform(gap, y_grid, log_correction)
                for j in range(i + 1, n_particles):
                    diff = (x[walker, j] - offsets[j]) - y_i
                    if diff <= 0.0 or not np.isfinite(diff):
                        valid = False
                        break
                    pair_log += np.log(diff)
                if not valid:
                    break
        if valid:
            value = -0.5 * alpha * gaussian_sum + pair_power * pair_log + gap_corr
            values[walker] = value
            finite[walker] = bool(np.isfinite(value))
        else:
            values[walker] = -np.inf
            finite[walker] = False
    return values, finite


def _gap_h_grad_lap_local_batch_python(
    x: FloatArray,
    offsets: FloatArray,
    y_grid: FloatArray,
    grad_correction: FloatArray,
    lap_correction: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    center: float,
    omega2: float,
    pair_power: float,
    n2_total_energy: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
    walkers, n_particles = x.shape
    grad = np.empty((walkers, n_particles), dtype=float)
    lap = np.empty((walkers, n_particles), dtype=float)
    local = np.empty(walkers, dtype=float)
    finite = np.empty(walkers, dtype=bool)
    y_max = float(y_grid[-1])
    for walker in range(walkers):
        row_ok = True
        for i in range(n_particles):
            y_i = x[walker, i] - offsets[i]
            grad_i = -alpha * (y_i - center)
            lap_i = -alpha
            for j in range(n_particles):
                if i == j:
                    continue
                diff = y_i - (x[walker, j] - offsets[j])
                inv = np.inf if diff == 0.0 else 1.0 / diff
                grad_i += pair_power * inv
                lap_i -= pair_power * inv * inv
            grad[walker, i] = grad_i
            lap[walker, i] = lap_i
            row_ok = row_ok and np.isfinite(grad_i) and np.isfinite(lap_i)
        for gap_index in range(n_particles - 1):
            gap = x[walker, gap_index + 1] - x[walker, gap_index]
            if not np.isfinite(gap) or gap <= rod_length or gap > y_max:
                row_ok = False
                break
            c1 = _interp_uniform(gap, y_grid, grad_correction)
            c2 = _interp_uniform(gap, y_grid, lap_correction)
            grad[walker, gap_index] -= c1
            grad[walker, gap_index + 1] += c1
            lap[walker, gap_index] += c2
            lap[walker, gap_index + 1] += c2
        if row_ok and n_particles == 2:
            local[walker] = n2_total_energy
        elif row_ok:
            trap_sum = np.sum((x[walker] - center) ** 2)
            local[walker] = -np.sum(lap[walker] + grad[walker] * grad[walker])
            local[walker] += 0.5 * omega2 * trap_sum
        else:
            local[walker] = np.nan
        finite[walker] = row_ok and np.isfinite(local[walker])
    return grad, lap, local, finite


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _interp_uniform_numba(x: float, grid: FloatArray, values: FloatArray) -> float:
        if x <= grid[0]:
            return values[0]
        last = grid.size - 1
        if x >= grid[last]:
            return values[last]
        position = (x - grid[0]) / (grid[1] - grid[0])
        left = int(position)
        if left < 0:
            left = 0
        if left >= last:
            left = last - 1
        t = position - left
        return (1.0 - t) * values[left] + t * values[left + 1]

    @njit(cache=True, fastmath=False)
    def _gap_h_log_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        y_grid: FloatArray,
        log_correction: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        pair_power: float,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        walkers, n_particles = x.shape
        values = np.empty(walkers, dtype=np.float64)
        finite = np.empty(walkers, dtype=np.bool_)
        y_max = y_grid[y_grid.size - 1]
        for walker in range(walkers):
            valid = True
            previous = 0.0
            for particle in range(n_particles):
                value = x[walker, particle]
                if not np.isfinite(value):
                    valid = False
                    break
                if particle > 0 and value - previous <= rod_length:
                    valid = False
                    break
                previous = value
            pair_log = 0.0
            gaussian_sum = 0.0
            gap_corr = 0.0
            if valid:
                for i in range(n_particles):
                    y_i = x[walker, i] - offsets[i]
                    gaussian_sum += (y_i - center) * (y_i - center)
                    if i < n_particles - 1:
                        gap = x[walker, i + 1] - x[walker, i]
                        if gap > y_max:
                            valid = False
                            break
                        gap_corr += _interp_uniform_numba(gap, y_grid, log_correction)
                    for j in range(i + 1, n_particles):
                        diff = (x[walker, j] - offsets[j]) - y_i
                        if diff <= 0.0 or not np.isfinite(diff):
                            valid = False
                            break
                        pair_log += np.log(diff)
                    if not valid:
                        break
            if valid:
                value = -0.5 * alpha * gaussian_sum + pair_power * pair_log + gap_corr
                values[walker] = value
                finite[walker] = np.isfinite(value)
            else:
                values[walker] = -np.inf
                finite[walker] = False
        return values, finite

    @njit(cache=True, fastmath=False)
    def _gap_h_grad_lap_local_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        y_grid: FloatArray,
        grad_correction: FloatArray,
        lap_correction: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        omega2: float,
        pair_power: float,
        n2_total_energy: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        walkers, n_particles = x.shape
        grad = np.empty((walkers, n_particles), dtype=np.float64)
        lap = np.empty((walkers, n_particles), dtype=np.float64)
        local = np.empty(walkers, dtype=np.float64)
        finite = np.empty(walkers, dtype=np.bool_)
        y_max = y_grid[y_grid.size - 1]
        for walker in range(walkers):
            row_ok = True
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
                    else:
                        inv = 1.0 / diff
                    grad_i += pair_power * inv
                    lap_i -= pair_power * inv * inv
                grad[walker, i] = grad_i
                lap[walker, i] = lap_i
                if not np.isfinite(grad_i) or not np.isfinite(lap_i):
                    row_ok = False
            for gap_index in range(n_particles - 1):
                gap = x[walker, gap_index + 1] - x[walker, gap_index]
                if not np.isfinite(gap) or gap <= rod_length or gap > y_max:
                    row_ok = False
                    break
                c1 = _interp_uniform_numba(gap, y_grid, grad_correction)
                c2 = _interp_uniform_numba(gap, y_grid, lap_correction)
                grad[walker, gap_index] -= c1
                grad[walker, gap_index + 1] += c1
                lap[walker, gap_index] += c2
                lap[walker, gap_index + 1] += c2
            if row_ok and n_particles == 2:
                local[walker] = n2_total_energy
            elif row_ok:
                trap_sum = 0.0
                local_sum = 0.0
                for particle in range(n_particles):
                    centered = x[walker, particle] - center
                    trap_sum += centered * centered
                    local_sum += (
                        lap[walker, particle]
                        + grad[walker, particle] * grad[walker, particle]
                    )
                local[walker] = -local_sum + 0.5 * omega2 * trap_sum
            else:
                local[walker] = np.nan
            finite[walker] = row_ok and np.isfinite(local[walker])
        return grad, lap, local, finite

else:

    def _gap_h_log_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        y_grid: FloatArray,
        log_correction: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        pair_power: float,
    ) -> tuple[FloatArray, NDArray[np.bool_]]:
        return _gap_h_log_batch_python(
            x,
            offsets,
            y_grid,
            log_correction,
            rod_length=rod_length,
            alpha=alpha,
            center=center,
            pair_power=pair_power,
        )

    def _gap_h_grad_lap_local_batch_numba(
        x: FloatArray,
        offsets: FloatArray,
        y_grid: FloatArray,
        grad_correction: FloatArray,
        lap_correction: FloatArray,
        rod_length: float,
        alpha: float,
        center: float,
        omega2: float,
        pair_power: float,
        n2_total_energy: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        return _gap_h_grad_lap_local_batch_python(
            x,
            offsets,
            y_grid,
            grad_correction,
            lap_correction,
            rod_length=rod_length,
            alpha=alpha,
            center=center,
            omega2=omega2,
            pair_power=pair_power,
            n2_total_energy=n2_total_energy,
        )
