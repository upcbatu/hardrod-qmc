from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics.numba_backend import NUMBA_AVAILABLE, njit
from hrdmc.wavefunctions.kernels.trapped_tg import (
    reduced_tg_grad_lap_local_batch,
    reduced_tg_log_batch,
    reduced_tg_relative_width_local_energy_batch,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_TAIL_SERIES_MAX_TERMS = 96
_TAIL_SERIES_REL_TOL = 5.0e-16


@dataclass(frozen=True)
class ContactTGSufficientStatistics:
    """Per-configuration statistics for an ``(alpha_relative, beta)`` scan."""

    base_local_energy: FloatArray
    internal_norm2: FloatArray
    correction_log_sum: FloatArray
    correction_lap_sum: FloatArray
    harmonic_contact_cross: FloatArray
    internal_contact_cross: FloatArray
    contact_gradient_norm2: FloatArray
    finite: BoolArray

    def log_amplitude_difference(
        self,
        *,
        relative_alpha: float,
        contact_beta: float,
        reference_relative_alpha: float,
        reference_contact_beta: float,
    ) -> FloatArray:
        return (
            -0.5 * (float(relative_alpha) - float(reference_relative_alpha)) * self.internal_norm2
            + (float(contact_beta) - float(reference_contact_beta)) * self.correction_log_sum
        )

    def local_energy(
        self,
        *,
        relative_alpha: float,
        contact_beta: float,
        omega: float,
        n_particles: int,
    ) -> FloatArray:
        delta = float(relative_alpha) - float(omega)
        beta = float(contact_beta)
        return (
            self.base_local_energy
            + 0.5 * delta * (int(n_particles) ** 2 - 1)
            - (float(omega) * delta + 0.5 * delta * delta) * self.internal_norm2
            - beta * (self.correction_lap_sum + self.harmonic_contact_cross)
            + beta * delta * self.internal_contact_cross
            - 0.5 * beta * beta * self.contact_gradient_norm2
        )


def contact_tg_backend_name() -> str:
    return "numba" if NUMBA_AVAILABLE else "python"


def contact_tg_log_batch(
    x: FloatArray,
    offsets: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    relative_alpha: float,
    center: float,
    pair_power: float,
    contact_beta: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[FloatArray, BoolArray]:
    base_log, base_finite = reduced_tg_log_batch(
        np.asarray(x, dtype=float),
        np.asarray(offsets, dtype=float),
        rod_length=float(rod_length),
        alpha=float(alpha),
        relative_alpha=float(relative_alpha),
        center=float(center),
        pair_power=float(pair_power),
    )
    if float(contact_beta) == 0.0:
        return base_log, base_finite
    correction_log, correction_finite = _contact_log_sum_batch(
        np.asarray(x, dtype=float),
        np.asarray(breakpoints, dtype=float),
        np.asarray(coefficients, dtype=float),
        base_finite,
        rod_length=float(rod_length),
        omega=float(omega),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        zero_correction=bool(zero_correction),
    )
    finite = base_finite & correction_finite
    values = base_log + float(contact_beta) * correction_log
    return np.where(finite, values, -np.inf), finite


def contact_tg_grad_lap_local_batch(
    x: FloatArray,
    offsets: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    *,
    rod_length: float,
    alpha: float,
    relative_alpha: float,
    center: float,
    omega: float,
    pair_power: float,
    contact_beta: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray]:
    positions = np.asarray(x, dtype=float)
    offset_values = np.asarray(offsets, dtype=float)
    grad, lap, _generic_local, base_finite = reduced_tg_grad_lap_local_batch(
        positions,
        offset_values,
        rod_length=float(rod_length),
        alpha=float(alpha),
        relative_alpha=float(relative_alpha),
        center=float(center),
        omega2=float(omega) ** 2,
        pair_power=float(pair_power),
    )
    base_local = reduced_tg_relative_width_local_energy_batch(
        positions,
        rod_length=float(rod_length),
        omega=float(omega),
        relative_alpha=float(relative_alpha),
    )
    if float(contact_beta) == 0.0:
        return grad, lap, np.where(base_finite, base_local, _generic_local), base_finite

    terms = _contact_terms_batch(
        positions,
        grad,
        offset_values,
        np.asarray(breakpoints, dtype=float),
        np.asarray(coefficients, dtype=float),
        base_finite,
        rod_length=float(rod_length),
        center=float(center),
        omega=float(omega),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        zero_correction=bool(zero_correction),
    )
    (
        _correction_log,
        correction_grad,
        correction_lap,
        _internal_norm2,
        correction_lap_sum,
        guide_contact_cross,
        _internal_contact_cross,
        contact_gradient_norm2,
        correction_finite,
    ) = terms
    beta = float(contact_beta)
    finite = base_finite & correction_finite
    corrected_grad = grad + beta * correction_grad
    corrected_lap = lap + beta * correction_lap
    local = (
        base_local
        - beta * correction_lap_sum
        - beta * guide_contact_cross
        - 0.5 * beta * beta * contact_gradient_norm2
    )
    finite &= np.isfinite(local)
    local = np.where(finite, local, np.nan)
    return corrected_grad, corrected_lap, local, finite


def contact_tg_sufficient_statistics_batch(
    x: FloatArray,
    offsets: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    *,
    rod_length: float,
    center: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> ContactTGSufficientStatistics:
    positions = np.asarray(x, dtype=float)
    offset_values = np.asarray(offsets, dtype=float)
    harmonic_grad, _lap, _local, base_finite = reduced_tg_grad_lap_local_batch(
        positions,
        offset_values,
        rod_length=float(rod_length),
        alpha=float(omega),
        relative_alpha=float(omega),
        center=float(center),
        omega2=float(omega) ** 2,
        pair_power=1.0,
    )
    terms = _contact_terms_batch(
        positions,
        harmonic_grad,
        offset_values,
        np.asarray(breakpoints, dtype=float),
        np.asarray(coefficients, dtype=float),
        base_finite,
        rod_length=float(rod_length),
        center=float(center),
        omega=float(omega),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        zero_correction=bool(zero_correction),
    )
    (
        correction_log,
        _correction_grad,
        _correction_lap,
        internal_norm2,
        correction_lap_sum,
        harmonic_contact_cross,
        internal_contact_cross,
        contact_gradient_norm2,
        correction_finite,
    ) = terms
    finite = base_finite & correction_finite
    base_local = reduced_tg_relative_width_local_energy_batch(
        positions,
        rod_length=float(rod_length),
        omega=float(omega),
        relative_alpha=float(omega),
    )
    return ContactTGSufficientStatistics(
        base_local_energy=np.where(finite, base_local, np.nan),
        internal_norm2=np.where(finite, internal_norm2, np.nan),
        correction_log_sum=np.where(finite, correction_log, np.nan),
        correction_lap_sum=np.where(finite, correction_lap_sum, np.nan),
        harmonic_contact_cross=np.where(finite, harmonic_contact_cross, np.nan),
        internal_contact_cross=np.where(finite, internal_contact_cross, np.nan),
        contact_gradient_norm2=np.where(finite, contact_gradient_norm2, np.nan),
        finite=finite,
    )


def _contact_log_sum_batch(
    x: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    base_finite: BoolArray,
    *,
    rod_length: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[FloatArray, BoolArray]:
    if NUMBA_AVAILABLE:
        return _contact_log_sum_batch_numba(
            x,
            breakpoints,
            coefficients,
            base_finite,
            float(rod_length),
            float(omega),
            float(tail_nu),
            float(tail_constant),
            bool(zero_correction),
        )
    return _contact_log_sum_batch_python(
        x,
        breakpoints,
        coefficients,
        base_finite,
        rod_length=float(rod_length),
        omega=float(omega),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        zero_correction=bool(zero_correction),
    )


def _contact_terms_batch(
    x: FloatArray,
    base_grad: FloatArray,
    offsets: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    base_finite: BoolArray,
    *,
    rod_length: float,
    center: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
]:
    if NUMBA_AVAILABLE:
        return _contact_terms_batch_numba(
            x,
            base_grad,
            offsets,
            breakpoints,
            coefficients,
            base_finite,
            float(rod_length),
            float(center),
            float(omega),
            float(tail_nu),
            float(tail_constant),
            bool(zero_correction),
        )
    return _contact_terms_batch_python(
        x,
        base_grad,
        offsets,
        breakpoints,
        coefficients,
        base_finite,
        rod_length=float(rod_length),
        center=float(center),
        omega=float(omega),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        zero_correction=bool(zero_correction),
    )


def _contact_log_sum_batch_python(
    x: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    base_finite: BoolArray,
    *,
    rod_length: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[FloatArray, BoolArray]:
    walkers, n_particles = x.shape
    log_sum = np.zeros(walkers, dtype=float)
    finite = np.asarray(base_finite, dtype=bool).copy()
    for walker in range(walkers):
        if not finite[walker]:
            continue
        for gap_index in range(n_particles - 1):
            free_gap = x[walker, gap_index + 1] - x[walker, gap_index] - rod_length
            if not np.isfinite(free_gap) or free_gap <= 0.0:
                finite[walker] = False
                break
            value, _first, _second = _evaluate_contact_python(
                free_gap,
                breakpoints,
                coefficients,
                rod_length=rod_length,
                omega=omega,
                tail_nu=tail_nu,
                tail_constant=tail_constant,
                zero_correction=zero_correction,
            )
            log_sum[walker] += value
    return log_sum, finite


def _contact_terms_batch_python(
    x: FloatArray,
    base_grad: FloatArray,
    offsets: FloatArray,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    base_finite: BoolArray,
    *,
    rod_length: float,
    center: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    BoolArray,
]:
    walkers, n_particles = x.shape
    correction_log = np.zeros(walkers, dtype=float)
    correction_grad = np.zeros_like(x, dtype=float)
    correction_lap = np.zeros_like(x, dtype=float)
    internal_norm2 = np.zeros(walkers, dtype=float)
    correction_lap_sum = np.zeros(walkers, dtype=float)
    guide_contact_cross = np.zeros(walkers, dtype=float)
    internal_contact_cross = np.zeros(walkers, dtype=float)
    contact_gradient_norm2 = np.zeros(walkers, dtype=float)
    finite = np.asarray(base_finite, dtype=bool).copy()
    for walker in range(walkers):
        if not finite[walker]:
            continue
        reduced = x[walker] - offsets - center
        com = float(np.mean(reduced))
        internal = reduced - com
        internal_norm2[walker] = float(np.sum(internal * internal))
        for gap_index in range(n_particles - 1):
            free_gap = x[walker, gap_index + 1] - x[walker, gap_index] - rod_length
            if not np.isfinite(free_gap) or free_gap <= 0.0:
                finite[walker] = False
                break
            value, first, second = _evaluate_contact_python(
                free_gap,
                breakpoints,
                coefficients,
                rod_length=rod_length,
                omega=omega,
                tail_nu=tail_nu,
                tail_constant=tail_constant,
                zero_correction=zero_correction,
            )
            correction_log[walker] += value
            correction_grad[walker, gap_index] -= first
            correction_grad[walker, gap_index + 1] += first
            correction_lap[walker, gap_index] += second
            correction_lap[walker, gap_index + 1] += second
            correction_lap_sum[walker] += second
        if finite[walker]:
            guide_contact_cross[walker] = float(np.dot(base_grad[walker], correction_grad[walker]))
            internal_contact_cross[walker] = float(np.dot(internal, correction_grad[walker]))
            contact_gradient_norm2[walker] = float(
                np.dot(correction_grad[walker], correction_grad[walker])
            )
    return (
        correction_log,
        correction_grad,
        correction_lap,
        internal_norm2,
        correction_lap_sum,
        guide_contact_cross,
        internal_contact_cross,
        contact_gradient_norm2,
        finite,
    )


def _evaluate_contact_python(
    free_gap: float,
    breakpoints: FloatArray,
    coefficients: FloatArray,
    *,
    rod_length: float,
    omega: float,
    tail_nu: float,
    tail_constant: float,
    zero_correction: bool,
) -> tuple[float, float, float]:
    if zero_correction:
        return 0.0, 0.0, 0.0
    if free_gap <= breakpoints[-1]:
        interval = int(np.searchsorted(breakpoints, free_gap, side="right") - 1)
        interval = max(0, min(interval, coefficients.shape[0] - 1))
        dx = free_gap - breakpoints[interval]
        c0, c1, c2, c3, c4, c5 = coefficients[interval]
        value = c0 + dx * (c1 + dx * (c2 + dx * (c3 + dx * (c4 + dx * c5))))
        first = c1 + dx * (2.0 * c2 + dx * (3.0 * c3 + dx * (4.0 * c4 + dx * 5.0 * c5)))
        second = 2.0 * c2 + dx * (6.0 * c3 + dx * (12.0 * c4 + dx * 20.0 * c5))
        return float(value), float(first), float(second)
    distance = rod_length + free_gap
    z = np.sqrt(omega) * distance
    log_series, log_series_first, log_series_second = (
        _parabolic_cylinder_tail_log_derivatives_python(tail_nu, z)
    )
    value = (
        tail_constant
        - 0.5 * omega * rod_length * free_gap
        + tail_nu * np.log(distance)
        - np.log(free_gap)
        + log_series
    )
    first = (
        -0.5 * omega * rod_length
        + tail_nu / distance
        - 1.0 / free_gap
        + np.sqrt(omega) * log_series_first
    )
    second = (
        -tail_nu / (distance * distance) + 1.0 / (free_gap * free_gap) + omega * log_series_second
    )
    return float(value), float(first), float(second)


def _parabolic_cylinder_tail_log_derivatives_python(
    nu: float,
    z: float,
) -> tuple[float, float, float]:
    total = 1.0
    total_first = 0.0
    total_second = 0.0
    term = 1.0
    best_magnitude = np.inf
    best_total = total
    best_first = total_first
    best_second = total_second
    for order in range(1, _TAIL_SERIES_MAX_TERMS + 1):
        left = nu - float(2 * order - 2)
        right = nu - float(2 * order - 1)
        term *= -(left * right) / (2.0 * float(order) * z * z)
        if not np.isfinite(term):
            break
        order_twice = float(2 * order)
        total += term
        total_first += -(order_twice / z) * term
        total_second += (order_twice * (order_twice + 1.0) / (z * z)) * term
        magnitude = abs(term)
        if magnitude < best_magnitude:
            best_magnitude = magnitude
            best_total = total
            best_first = total_first
            best_second = total_second
        if magnitude <= _TAIL_SERIES_REL_TOL * max(abs(total), 1.0):
            best_total = total
            best_first = total_first
            best_second = total_second
            break
    log_first = best_first / best_total
    return (
        float(np.log(best_total)),
        float(log_first),
        float(best_second / best_total - log_first * log_first),
    )


if NUMBA_AVAILABLE:

    @njit(fastmath=False)
    def _parabolic_cylinder_tail_log_derivatives_numba(
        nu: float,
        z: float,
    ) -> tuple[float, float, float]:
        total = 1.0
        total_first = 0.0
        total_second = 0.0
        term = 1.0
        best_magnitude = np.inf
        best_total = total
        best_first = total_first
        best_second = total_second
        for order in range(1, _TAIL_SERIES_MAX_TERMS + 1):
            left = nu - float(2 * order - 2)
            right = nu - float(2 * order - 1)
            term *= -(left * right) / (2.0 * float(order) * z * z)
            if not np.isfinite(term):
                break
            order_twice = float(2 * order)
            total += term
            total_first += -(order_twice / z) * term
            total_second += (order_twice * (order_twice + 1.0) / (z * z)) * term
            magnitude = abs(term)
            if magnitude < best_magnitude:
                best_magnitude = magnitude
                best_total = total
                best_first = total_first
                best_second = total_second
            if magnitude <= _TAIL_SERIES_REL_TOL * max(abs(total), 1.0):
                best_total = total
                best_first = total_first
                best_second = total_second
                break
        log_first = best_first / best_total
        return (
            np.log(best_total),
            log_first,
            best_second / best_total - log_first * log_first,
        )

    @njit(fastmath=False)
    def _evaluate_contact_numba(
        free_gap: float,
        breakpoints: FloatArray,
        coefficients: FloatArray,
        rod_length: float,
        omega: float,
        tail_nu: float,
        tail_constant: float,
        zero_correction: bool,
    ) -> tuple[float, float, float]:
        if zero_correction:
            return 0.0, 0.0, 0.0
        if free_gap <= breakpoints[breakpoints.size - 1]:
            lo = 0
            hi = breakpoints.size
            while lo < hi:
                mid = (lo + hi) // 2
                if breakpoints[mid] <= free_gap:
                    lo = mid + 1
                else:
                    hi = mid
            interval = lo - 1
            if interval < 0:
                interval = 0
            if interval >= coefficients.shape[0]:
                interval = coefficients.shape[0] - 1
            dx = free_gap - breakpoints[interval]
            c0 = coefficients[interval, 0]
            c1 = coefficients[interval, 1]
            c2 = coefficients[interval, 2]
            c3 = coefficients[interval, 3]
            c4 = coefficients[interval, 4]
            c5 = coefficients[interval, 5]
            value = c0 + dx * (c1 + dx * (c2 + dx * (c3 + dx * (c4 + dx * c5))))
            first = c1 + dx * (2.0 * c2 + dx * (3.0 * c3 + dx * (4.0 * c4 + dx * 5.0 * c5)))
            second = 2.0 * c2 + dx * (6.0 * c3 + dx * (12.0 * c4 + dx * 20.0 * c5))
            return value, first, second
        distance = rod_length + free_gap
        z = np.sqrt(omega) * distance
        log_series, log_series_first, log_series_second = (
            _parabolic_cylinder_tail_log_derivatives_numba(tail_nu, z)
        )
        value = (
            tail_constant
            - 0.5 * omega * rod_length * free_gap
            + tail_nu * np.log(distance)
            - np.log(free_gap)
            + log_series
        )
        first = (
            -0.5 * omega * rod_length
            + tail_nu / distance
            - 1.0 / free_gap
            + np.sqrt(omega) * log_series_first
        )
        second = (
            -tail_nu / (distance * distance)
            + 1.0 / (free_gap * free_gap)
            + omega * log_series_second
        )
        return value, first, second

    @njit(fastmath=False)
    def _contact_log_sum_batch_numba(
        x: FloatArray,
        breakpoints: FloatArray,
        coefficients: FloatArray,
        base_finite: BoolArray,
        rod_length: float,
        omega: float,
        tail_nu: float,
        tail_constant: float,
        zero_correction: bool,
    ) -> tuple[FloatArray, BoolArray]:
        walkers, n_particles = x.shape
        log_sum = np.zeros(walkers, dtype=np.float64)
        finite = base_finite.copy()
        for walker in range(walkers):
            if not finite[walker]:
                continue
            for gap_index in range(n_particles - 1):
                free_gap = x[walker, gap_index + 1] - x[walker, gap_index] - rod_length
                if not np.isfinite(free_gap) or free_gap <= 0.0:
                    finite[walker] = False
                    break
                value, _first, _second = _evaluate_contact_numba(
                    free_gap,
                    breakpoints,
                    coefficients,
                    rod_length,
                    omega,
                    tail_nu,
                    tail_constant,
                    zero_correction,
                )
                log_sum[walker] += value
        return log_sum, finite

    @njit(fastmath=False)
    def _contact_terms_batch_numba(
        x: FloatArray,
        base_grad: FloatArray,
        offsets: FloatArray,
        breakpoints: FloatArray,
        coefficients: FloatArray,
        base_finite: BoolArray,
        rod_length: float,
        center: float,
        omega: float,
        tail_nu: float,
        tail_constant: float,
        zero_correction: bool,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        BoolArray,
    ]:
        walkers, n_particles = x.shape
        correction_log = np.zeros(walkers, dtype=np.float64)
        correction_grad = np.zeros((walkers, n_particles), dtype=np.float64)
        correction_lap = np.zeros((walkers, n_particles), dtype=np.float64)
        internal_norm2 = np.zeros(walkers, dtype=np.float64)
        correction_lap_sum = np.zeros(walkers, dtype=np.float64)
        guide_contact_cross = np.zeros(walkers, dtype=np.float64)
        internal_contact_cross = np.zeros(walkers, dtype=np.float64)
        contact_gradient_norm2 = np.zeros(walkers, dtype=np.float64)
        finite = base_finite.copy()
        internal = np.empty(n_particles, dtype=np.float64)
        for walker in range(walkers):
            if not finite[walker]:
                continue
            reduced_sum = 0.0
            for particle in range(n_particles):
                reduced_sum += x[walker, particle] - offsets[particle] - center
            com = reduced_sum / n_particles
            for particle in range(n_particles):
                value = x[walker, particle] - offsets[particle] - center - com
                internal[particle] = value
                internal_norm2[walker] += value * value
            for gap_index in range(n_particles - 1):
                free_gap = x[walker, gap_index + 1] - x[walker, gap_index] - rod_length
                if not np.isfinite(free_gap) or free_gap <= 0.0:
                    finite[walker] = False
                    break
                value, first, second = _evaluate_contact_numba(
                    free_gap,
                    breakpoints,
                    coefficients,
                    rod_length,
                    omega,
                    tail_nu,
                    tail_constant,
                    zero_correction,
                )
                correction_log[walker] += value
                correction_grad[walker, gap_index] -= first
                correction_grad[walker, gap_index + 1] += first
                correction_lap[walker, gap_index] += second
                correction_lap[walker, gap_index + 1] += second
                correction_lap_sum[walker] += second
            if finite[walker]:
                for particle in range(n_particles):
                    contact_value = correction_grad[walker, particle]
                    guide_contact_cross[walker] += base_grad[walker, particle] * contact_value
                    internal_contact_cross[walker] += internal[particle] * contact_value
                    contact_gradient_norm2[walker] += contact_value * contact_value
        return (
            correction_log,
            correction_grad,
            correction_lap,
            internal_norm2,
            correction_lap_sum,
            guide_contact_cross,
            internal_contact_cross,
            contact_gradient_norm2,
            finite,
        )

else:

    def _contact_log_sum_batch_numba(
        x: FloatArray,
        breakpoints: FloatArray,
        coefficients: FloatArray,
        base_finite: BoolArray,
        rod_length: float,
        omega: float,
        tail_nu: float,
        tail_constant: float,
        zero_correction: bool,
    ) -> tuple[FloatArray, BoolArray]:
        return _contact_log_sum_batch_python(
            x,
            breakpoints,
            coefficients,
            base_finite,
            rod_length=rod_length,
            omega=omega,
            tail_nu=tail_nu,
            tail_constant=tail_constant,
            zero_correction=zero_correction,
        )

    def _contact_terms_batch_numba(
        x: FloatArray,
        base_grad: FloatArray,
        offsets: FloatArray,
        breakpoints: FloatArray,
        coefficients: FloatArray,
        base_finite: BoolArray,
        rod_length: float,
        center: float,
        omega: float,
        tail_nu: float,
        tail_constant: float,
        zero_correction: bool,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        BoolArray,
    ]:
        return _contact_terms_batch_python(
            x,
            base_grad,
            offsets,
            breakpoints,
            coefficients,
            base_finite,
            rod_length=rod_length,
            center=center,
            omega=omega,
            tail_nu=tail_nu,
            tail_constant=tail_constant,
            zero_correction=zero_correction,
        )
