from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.analysis.timeseries import integrated_autocorrelation_time

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CorrelatedErrorEstimate:
    """Mean-error estimate for a correlated scalar time series."""

    method: str
    tau_int_samples: float
    tau_int_tau: float
    effective_sample_size: float
    stderr_mean: float
    stderr_error: float
    bandwidth: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "method": self.method,
            "tau_int_samples": self.tau_int_samples,
            "tau_int_tau": self.tau_int_tau,
            "effective_sample_size": self.effective_sample_size,
            "stderr_mean": self.stderr_mean,
            "stderr_error": self.stderr_error,
            "bandwidth": self.bandwidth,
        }


@dataclass(frozen=True)
class TriangulatedErrorResult:
    """Conservative correlated-error estimate from multiple estimators."""

    estimates: tuple[CorrelatedErrorEstimate, ...]
    conservative_stderr: float
    overlap_pair_count: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "conservative_stderr": self.conservative_stderr,
            "overlap_pair_count": self.overlap_pair_count,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
        }


TRIANGULATED_STATUS = "TRIANGULATED_2_OF_3"
DISAGREE_STATUS = "DISAGREE_HONEST_LARGE"
UNAVAILABLE_STATUS = "CORRELATED_ERROR_UNAVAILABLE"


def triangulated_error_estimate(
    values: FloatArray,
    *,
    trace_spacing_tau: float = 1.0,
    overlap_sigma: float = 1.0,
) -> TriangulatedErrorResult:
    """Estimate a correlated-trace mean error by Sokal/Geyer/HAC triangulation.

    The returned stderr is always the maximum finite stderr among the three
    estimators. A ``TRIANGULATED_2_OF_3`` status means at least one pair agrees
    within its estimated one-sigma stderr uncertainty. Disagreement is not
    accepted as a precise result; it is reported as an honest large-error
    precision warning rather than silently shrinking the error bar.
    """

    arr = _finite_1d(values)
    estimates = (
        sokal_error_estimate(arr, trace_spacing_tau=trace_spacing_tau),
        geyer_error_estimate(arr, trace_spacing_tau=trace_spacing_tau),
        hac_flat_top_error_estimate(arr, trace_spacing_tau=trace_spacing_tau),
    )
    finite_stderr = [
        estimate.stderr_mean
        for estimate in estimates
        if np.isfinite(estimate.stderr_mean)
    ]
    if not finite_stderr:
        return TriangulatedErrorResult(
            estimates=estimates,
            conservative_stderr=float("nan"),
            overlap_pair_count=0,
            status=UNAVAILABLE_STATUS,
        )
    overlap_count = _overlap_pair_count(estimates, sigma=overlap_sigma)
    return TriangulatedErrorResult(
        estimates=estimates,
        conservative_stderr=float(max(finite_stderr)),
        overlap_pair_count=overlap_count,
        status=TRIANGULATED_STATUS if overlap_count >= 1 else DISAGREE_STATUS,
    )


def sokal_error_estimate(
    values: FloatArray,
    *,
    trace_spacing_tau: float = 1.0,
) -> CorrelatedErrorEstimate:
    """Sokal-window integrated-autocorrelation error estimate."""

    arr = _finite_1d(values)
    acf = integrated_autocorrelation_time(arr)
    variance = float(np.var(arr, ddof=1)) if arr.size > 1 else float("nan")
    return _estimate_from_tau(
        method="sokal",
        tau_int=float(acf.tau_int_samples),
        spacing=float(trace_spacing_tau),
        n=arr.size,
        variance=variance,
        bandwidth=None,
    )


def geyer_error_estimate(
    values: FloatArray,
    *,
    trace_spacing_tau: float = 1.0,
) -> CorrelatedErrorEstimate:
    """Geyer initial-positive/initial-monotone sequence error estimate."""

    arr = _finite_1d(values)
    n = arr.size
    if n < 4:
        return _estimate_from_tau(
            method="geyer_initial_sequence",
            tau_int=float("nan"),
            spacing=float(trace_spacing_tau),
            n=n,
            variance=float("nan"),
            bandwidth=None,
        )
    gamma = _autocovariances_fft(arr)
    gamma0 = float(gamma[0])
    if gamma0 <= 0.0 or not np.isfinite(gamma0):
        return _estimate_from_tau(
            method="geyer_initial_sequence",
            tau_int=0.5,
            spacing=float(trace_spacing_tau),
            n=n,
            variance=0.0,
            bandwidth=None,
        )
    pair_count = (gamma.size - 1) // 2
    pairs = np.asarray(
        [gamma[2 * k] + gamma[2 * k + 1] for k in range(pair_count)],
        dtype=float,
    )
    positive: list[float] = []
    for value in pairs:
        if not np.isfinite(value) or value <= 0.0:
            break
        positive.append(float(value))
    if positive:
        monotone = np.minimum.accumulate(np.asarray(positive, dtype=float))
        tau_int = max(0.5, float(-0.5 + np.sum(monotone) / gamma0))
    else:
        tau_int = 0.5
    return _estimate_from_tau(
        method="geyer_initial_sequence",
        tau_int=tau_int,
        spacing=float(trace_spacing_tau),
        n=n,
        variance=float(np.var(arr, ddof=1)),
        bandwidth=None,
    )


def hac_flat_top_error_estimate(
    values: FloatArray,
    *,
    trace_spacing_tau: float = 1.0,
) -> CorrelatedErrorEstimate:
    """Flat-top HAC spectral-density-at-zero error estimate."""

    arr = _finite_1d(values)
    n = arr.size
    if n < 4:
        return _estimate_from_tau(
            method="flat_top_hac",
            tau_int=float("nan"),
            spacing=float(trace_spacing_tau),
            n=n,
            variance=float("nan"),
            bandwidth=float("nan"),
        )
    gamma = _autocovariances_fft(arr)
    gamma0 = float(gamma[0])
    if gamma0 <= 0.0 or not np.isfinite(gamma0):
        return _estimate_from_tau(
            method="flat_top_hac",
            tau_int=0.5,
            spacing=float(trace_spacing_tau),
            n=n,
            variance=0.0,
            bandwidth=1.0,
        )
    bandwidth = _hac_plugin_bandwidth(arr)
    lags = np.arange(1, min(bandwidth, gamma.size - 1) + 1, dtype=int)
    weights = _flat_top_kernel(lags / float(bandwidth))
    spectral0 = gamma0 + 2.0 * float(np.sum(weights * gamma[lags]))
    spectral0 = max(spectral0, gamma0 / max(n, 1))
    tau_int = max(0.5, spectral0 / (2.0 * gamma0))
    return _estimate_from_tau(
        method="flat_top_hac",
        tau_int=tau_int,
        spacing=float(trace_spacing_tau),
        n=n,
        variance=float(np.var(arr, ddof=1)),
        bandwidth=float(bandwidth),
    )


def _estimate_from_tau(
    *,
    method: str,
    tau_int: float,
    spacing: float,
    n: int,
    variance: float,
    bandwidth: float | None,
) -> CorrelatedErrorEstimate:
    if not np.isfinite(tau_int) or not np.isfinite(variance) or n <= 0:
        neff = float("nan")
        stderr = float("nan")
    else:
        tau = max(float(tau_int), 0.5)
        neff = float(n / (2.0 * tau))
        stderr = float(math.sqrt(max(variance, 0.0) * 2.0 * tau / n))
    if np.isfinite(stderr) and np.isfinite(neff) and neff > 1.0:
        stderr_error = float(stderr / math.sqrt(2.0 * (neff - 1.0)))
    else:
        stderr_error = float("nan")
    return CorrelatedErrorEstimate(
        method=method,
        tau_int_samples=float(tau_int),
        tau_int_tau=float(tau_int * spacing),
        effective_sample_size=float(neff),
        stderr_mean=float(stderr),
        stderr_error=stderr_error,
        bandwidth=bandwidth,
    )


def _overlap_pair_count(
    estimates: tuple[CorrelatedErrorEstimate, ...],
    *,
    sigma: float,
) -> int:
    count = 0
    for i, left in enumerate(estimates):
        for right in estimates[i + 1 :]:
            values = [
                left.stderr_mean,
                right.stderr_mean,
                left.stderr_error,
                right.stderr_error,
            ]
            if not np.all(np.isfinite(values)):
                continue
            combined = math.sqrt(left.stderr_error**2 + right.stderr_error**2)
            delta = abs(left.stderr_mean - right.stderr_mean)
            if combined == 0.0 and delta == 0.0:
                count += 1
            elif delta <= sigma * combined:
                count += 1
    return count


def _autocovariances_fft(values: FloatArray) -> FloatArray:
    arr = _finite_1d(values)
    centered = arr - float(np.mean(arr))
    n = centered.size
    padded = np.zeros(2 * n, dtype=float)
    padded[:n] = centered
    spectrum = np.fft.rfft(padded)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded.size)[:n]
    return raw / np.arange(n, 0, -1, dtype=float)


def _hac_plugin_bandwidth(values: FloatArray) -> int:
    arr = _finite_1d(values)
    n = arr.size
    if n < 4:
        return 1
    rho = min(abs(_lag1_correlation(arr)), 0.98)
    plugin = (4.0 * rho * rho / max((1.0 - rho) ** 4, 1e-12)) * n
    return int(np.clip(math.ceil(1.1447 * plugin ** (1.0 / 3.0)), 2, max(2, n // 3)))


def _lag1_correlation(values: FloatArray) -> float:
    arr = _finite_1d(values)
    if arr.size < 3:
        return 0.0
    left = arr[:-1] - float(np.mean(arr[:-1]))
    right = arr[1:] - float(np.mean(arr[1:]))
    denom = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denom) if denom > 0.0 else 0.0


def _flat_top_kernel(values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)
    abs_arr = np.abs(arr)
    out[abs_arr <= 0.5] = 1.0
    taper = (abs_arr > 0.5) & (abs_arr <= 1.0)
    out[taper] = 2.0 * (1.0 - abs_arr[taper])
    return out


def _finite_1d(values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]
