from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import pbdv

from hrdmc.systems.gap_h_pair_table import build_gap_h_pair_ground_state

FloatArray = NDArray[np.float64]

_RELIABLE_GROUND_STATE_FRACTION = 1.0e-12
_TAIL_SERIES_MAX_TERMS = 96
_TAIL_SERIES_REL_TOL = 5.0e-16


@dataclass(frozen=True)
class N2GapContactCorrection:
    """Contact-regular N=2 gap correction for a reduced-TG guide.

    The polynomial coefficients use ascending powers of the displacement from
    the left edge of each interval.  Values beyond ``breakpoints[-1]`` use the
    stable decaying parabolic-cylinder tail expansion.  The two
    representations are joined with matching value, first derivative, and
    second derivative.
    """

    rod_length: float
    omega: float
    relative_energy: float
    contact_curvature: float
    breakpoints: FloatArray
    coefficients: FloatArray
    tail_nu: float
    tail_constant: float
    numerical_tail_start: float
    zero_correction: bool = False

    @property
    def n2_total_energy(self) -> float:
        return float(self.relative_energy + 0.5 * self.omega)

    def evaluate(self, free_gaps: FloatArray | float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate ``c(g)``, ``c'(g)``, and ``c''(g)`` consistently."""

        gaps = np.asarray(free_gaps, dtype=float)
        if np.any(~np.isfinite(gaps)) or np.any(gaps < 0.0):
            raise ValueError("free gaps must be finite and non-negative")
        if self.zero_correction:
            zeros = np.zeros_like(gaps, dtype=float)
            return zeros, zeros.copy(), zeros.copy()

        flat = gaps.reshape(-1)
        values = np.empty_like(flat)
        first = np.empty_like(flat)
        second = np.empty_like(flat)
        final_breakpoint = float(self.breakpoints[-1])
        for index, gap in enumerate(flat):
            if gap <= final_breakpoint:
                interval = int(np.searchsorted(self.breakpoints, gap, side="right") - 1)
                interval = max(0, min(interval, self.coefficients.shape[0] - 1))
                dx = float(gap - self.breakpoints[interval])
                values[index], first[index], second[index] = _evaluate_polynomial(
                    self.coefficients[interval],
                    dx,
                )
            else:
                values[index], first[index], second[index] = _tail_values(
                    float(gap),
                    rod_length=self.rod_length,
                    omega=self.omega,
                    nu=self.tail_nu,
                    constant=self.tail_constant,
                )
        return (
            values.reshape(gaps.shape),
            first.reshape(gaps.shape),
            second.reshape(gaps.shape),
        )


@lru_cache(maxsize=32)
def build_n2_gap_contact_correction(
    *,
    rod_length: float,
    omega: float,
    grid_points: int = 700,
    y_max: float | None = None,
) -> N2GapContactCorrection:
    """Build the exact-pair correction relative to the harmonic TG gap factor.

    For ``A=0`` the exact relative state is the harmonic TG factor, so the
    correction is returned as exactly zero.  For ``A>0``, the finite-difference
    ground state supplies the resolved interior and the known decaying
    parabolic-cylinder form supplies the far tail.
    """

    if rod_length < 0.0 or not math.isfinite(rod_length):
        raise ValueError("rod_length must be finite and non-negative")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    if math.isclose(rod_length, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        breakpoints = _readonly(np.asarray([0.0, 1.0], dtype=float))
        coefficients = _readonly(np.zeros((1, 6), dtype=float))
        return N2GapContactCorrection(
            rod_length=0.0,
            omega=float(omega),
            relative_energy=1.5 * float(omega),
            contact_curvature=0.0,
            breakpoints=breakpoints,
            coefficients=coefficients,
            tail_nu=1.0,
            tail_constant=0.0,
            numerical_tail_start=0.0,
            zero_correction=True,
        )

    pair = build_gap_h_pair_ground_state(
        rod_length=float(rod_length),
        omega=float(omega),
        grid_points=int(grid_points),
        y_max=y_max,
    )
    free_gap = np.asarray(pair.y_grid - rod_length, dtype=float)
    nu = _relative_parabolic_index(
        rod_length=float(rod_length),
        omega=float(omega),
        energy_guess=float(pair.relative_energy),
    )
    relative_energy = float(omega) * (nu + 0.5)
    z = math.sqrt(float(omega)) * (float(rod_length) + free_gap)
    ground_state, ground_state_z_derivative = pbdv(nu, z)
    ground_state = np.asarray(ground_state, dtype=float)
    ground_state_z_derivative = np.asarray(ground_state_z_derivative, dtype=float)
    boundary_value, boundary_z_derivative = pbdv(
        nu,
        math.sqrt(float(omega)) * float(rod_length),
    )
    del boundary_value
    boundary_derivative = math.sqrt(float(omega)) * float(boundary_z_derivative)
    if not math.isfinite(boundary_derivative) or boundary_derivative == 0.0:
        raise RuntimeError("invalid exact-pair contact derivative")
    maximum = float(np.max(np.abs(ground_state)))
    reliable = np.flatnonzero(
        np.isfinite(ground_state)
        & np.isfinite(ground_state_z_derivative)
        & (np.abs(ground_state) > maximum * _RELIABLE_GROUND_STATE_FRACTION)
        & (ground_state * boundary_derivative > 0.0)
    )
    if reliable.size < 8 or reliable[0] != 0:
        raise RuntimeError("N=2 ground-state table has insufficient reliable contact support")
    final_reliable = int(reliable[-1])
    free_gap = free_gap[: final_reliable + 1]
    ground_state = ground_state[: final_reliable + 1]
    ground_state_z_derivative = ground_state_z_derivative[: final_reliable + 1]

    contact_curvature = (
        float(omega) ** 2 * float(rod_length) ** 2 / 4.0 - relative_energy
    ) / 3.0 + 0.5 * float(omega)
    contact_constant = math.log(abs(boundary_derivative))
    correction_values = (
        np.log(np.abs(ground_state))
        - np.log(free_gap)
        + 0.25 * float(omega) * free_gap * free_gap
        - contact_constant
    )
    distance = float(rod_length) + free_gap
    log_ground_first = math.sqrt(float(omega)) * ground_state_z_derivative / ground_state
    correction_first = log_ground_first - 1.0 / free_gap + 0.5 * float(omega) * free_gap
    correction_second = (
        float(omega) ** 2 * distance * distance / 4.0
        - relative_energy
        - log_ground_first * log_ground_first
        + 1.0 / (free_gap * free_gap)
        + 0.5 * float(omega)
    )

    first_gap = float(free_gap[0])
    contact_end = (
        float(correction_values[0]),
        float(correction_first[0]),
        float(correction_second[0]),
    )
    interval_coefficients: list[FloatArray] = [
        _quintic_hermite_coefficients(
            0.0,
            first_gap,
            0.0,
            0.0,
            contact_curvature,
            *contact_end,
        )
    ]
    for interval in range(free_gap.size - 1):
        interval_coefficients.append(
            _quintic_hermite_coefficients(
                float(free_gap[interval]),
                float(free_gap[interval + 1]),
                float(correction_values[interval]),
                float(correction_first[interval]),
                float(correction_second[interval]),
                float(correction_values[interval + 1]),
                float(correction_first[interval + 1]),
                float(correction_second[interval + 1]),
            )
        )

    numerical_tail_start = float(free_gap[-1])
    numerical_tail_values = (
        float(correction_values[-1]),
        float(correction_first[-1]),
        float(correction_second[-1]),
    )
    tail_nu = nu
    unshifted_tail_at_start = _tail_values(
        numerical_tail_start,
        rod_length=float(rod_length),
        omega=float(omega),
        nu=tail_nu,
        constant=0.0,
    )[0]
    tail_constant = numerical_tail_values[0] - unshifted_tail_at_start
    tail_at_start = _tail_values(
        numerical_tail_start,
        rod_length=float(rod_length),
        omega=float(omega),
        nu=tail_nu,
        constant=tail_constant,
    )
    if not np.allclose(
        np.asarray(tail_at_start[1:]),
        np.asarray(numerical_tail_values[1:]),
        rtol=2.0e-10,
        atol=2.0e-10,
    ):
        raise RuntimeError("stable parabolic-cylinder tail does not match the resolved N=2 table")
    tail_blend_end = numerical_tail_start + max(1.0, 8.0 * float(pair.dy))
    tail_end_values = _tail_values(
        tail_blend_end,
        rod_length=float(rod_length),
        omega=float(omega),
        nu=tail_nu,
        constant=tail_constant,
    )
    interval_coefficients.append(
        _quintic_hermite_coefficients(
            numerical_tail_start,
            tail_blend_end,
            *numerical_tail_values,
            *tail_end_values,
        )
    )

    breakpoints = np.concatenate(
        (
            np.asarray([0.0], dtype=float),
            free_gap,
            np.asarray([tail_blend_end], dtype=float),
        )
    )
    coefficients = np.vstack(interval_coefficients)
    if coefficients.shape[0] != breakpoints.size - 1:
        raise RuntimeError("invalid contact-correction interval construction")
    return N2GapContactCorrection(
        rod_length=float(rod_length),
        omega=float(omega),
        relative_energy=relative_energy,
        contact_curvature=float(contact_curvature),
        breakpoints=_readonly(breakpoints),
        coefficients=_readonly(coefficients),
        tail_nu=float(tail_nu),
        tail_constant=float(tail_constant),
        numerical_tail_start=numerical_tail_start,
        zero_correction=False,
    )


def _relative_parabolic_index(
    *,
    rod_length: float,
    omega: float,
    energy_guess: float,
) -> float:
    z_contact = math.sqrt(float(omega)) * float(rod_length)
    guess = float(energy_guess) / float(omega) - 0.5

    def boundary_value(index: float) -> float:
        return float(pbdv(float(index), z_contact)[0])

    half_width = 0.05
    lower = guess - half_width
    upper = guess + half_width
    lower_value = boundary_value(lower)
    upper_value = boundary_value(upper)
    for _attempt in range(8):
        if (
            math.isfinite(lower_value)
            and math.isfinite(upper_value)
            and lower_value * upper_value <= 0.0
        ):
            break
        half_width *= 1.75
        lower = guess - half_width
        upper = guess + half_width
        lower_value = boundary_value(lower)
        upper_value = boundary_value(upper)
    else:
        raise RuntimeError("failed to bracket the exact-pair parabolic index")
    root = brentq(
        boundary_value,
        lower,
        upper,
        xtol=np.float64(5.0e-14),
        rtol=np.float64(1.0e-14),
    )
    index = float(cast(float, root))
    nearest_integer = round(index)
    if abs(index - nearest_integer) < 1.0e-8:
        index = float(nearest_integer)
    return index


def _quintic_hermite_coefficients(
    x0: float,
    x1: float,
    value0: float,
    first0: float,
    second0: float,
    value1: float,
    first1: float,
    second1: float,
) -> FloatArray:
    width = float(x1 - x0)
    if width <= 0.0:
        raise ValueError("quintic interval must have positive width")
    known = np.asarray([value0, first0, 0.5 * second0], dtype=float)
    rhs = np.asarray(
        [
            value1 - (known[0] + known[1] * width + known[2] * width**2),
            first1 - (known[1] + 2.0 * known[2] * width),
            second1 - 2.0 * known[2],
        ],
        dtype=float,
    )
    matrix = np.asarray(
        [
            [width**3, width**4, width**5],
            [3.0 * width**2, 4.0 * width**3, 5.0 * width**4],
            [6.0 * width, 12.0 * width**2, 20.0 * width**3],
        ],
        dtype=float,
    )
    high = np.linalg.solve(matrix, rhs)
    return np.concatenate((known, high))


def _evaluate_polynomial(coefficients: FloatArray, dx: float) -> tuple[float, float, float]:
    c0, c1, c2, c3, c4, c5 = (float(value) for value in coefficients)
    value = c0 + dx * (c1 + dx * (c2 + dx * (c3 + dx * (c4 + dx * c5))))
    first = c1 + dx * (2.0 * c2 + dx * (3.0 * c3 + dx * (4.0 * c4 + dx * 5.0 * c5)))
    second = 2.0 * c2 + dx * (6.0 * c3 + dx * (12.0 * c4 + dx * 20.0 * c5))
    return value, first, second


def _tail_values(
    free_gap: float,
    *,
    rod_length: float,
    omega: float,
    nu: float,
    constant: float,
) -> tuple[float, float, float]:
    distance = float(rod_length + free_gap)
    sqrt_omega = math.sqrt(float(omega))
    z = sqrt_omega * distance
    log_series, log_series_first, log_series_second = _parabolic_cylinder_tail_log_derivatives(
        float(nu), z
    )
    value = (
        float(constant)
        - 0.5 * float(omega) * float(rod_length) * float(free_gap)
        + float(nu) * math.log(distance)
        - math.log(float(free_gap))
        + log_series
    )
    first = (
        -0.5 * float(omega) * float(rod_length)
        + float(nu) / distance
        - 1.0 / float(free_gap)
        + sqrt_omega * log_series_first
    )
    second = (
        -float(nu) / (distance * distance)
        + 1.0 / (float(free_gap) ** 2)
        + float(omega) * log_series_second
    )
    return value, first, second


def _parabolic_cylinder_tail_log_derivatives(
    nu: float,
    z: float,
) -> tuple[float, float, float]:
    """Return log derivatives of the stable decaying-tail correction series.

    ``D_nu(z)`` is represented as ``exp(-z^2/4) * z^nu * S_nu(z)``.
    The optimally truncated inverse-power series for ``S_nu`` retains the
    smallest available term.  The builder verifies its value-independent
    first and second log derivatives against the resolved parabolic-cylinder
    table before the tail is admitted.
    """

    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("parabolic-cylinder tail coordinate must be positive and finite")
    total = 1.0
    total_first = 0.0
    total_second = 0.0
    term = 1.0
    best_magnitude = float("inf")
    best = (total, total_first, total_second)
    for order in range(1, _TAIL_SERIES_MAX_TERMS + 1):
        left = float(nu) - float(2 * order - 2)
        right = float(nu) - float(2 * order - 1)
        term *= -(left * right) / (2.0 * float(order) * z * z)
        if not math.isfinite(term):
            break
        order_twice = float(2 * order)
        total += term
        total_first += -(order_twice / z) * term
        total_second += (order_twice * (order_twice + 1.0) / (z * z)) * term
        magnitude = abs(term)
        if magnitude < best_magnitude:
            best_magnitude = magnitude
            best = (total, total_first, total_second)
        if magnitude <= _TAIL_SERIES_REL_TOL * max(abs(total), 1.0):
            best = (total, total_first, total_second)
            break

    series, series_first, series_second = best
    if not math.isfinite(series) or series <= 0.0:
        raise RuntimeError("stable parabolic-cylinder tail series is unavailable")
    log_first = series_first / series
    return (
        math.log(series),
        log_first,
        series_second / series - log_first * log_first,
    )


def _readonly(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array
