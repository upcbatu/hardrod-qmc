from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.statistics.blocking import blocking_standard_error

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class SlopeResult:
    slope: float
    slope_stderr: float
    slope_abs_z: float
    compatible_with_zero: bool
@dataclass(frozen=True)
class AutocorrelationResult:
    tau_int_samples: float
    window_lag: int | None
    effective_independent_samples: float
    reason: str
    acf: FloatArray
@dataclass(frozen=True)
class TraceStationarityResult:
    point_count: int
    slope: SlopeResult
    autocorrelation: AutocorrelationResult
    slope_stderr_autocorr_adjusted: float
    slope_z_autocorr_adjusted: float
    first_last_quarter_z: float
    late_cumulative_z: float
    first_last_blocking_z: float
    spread_blocking_z: float
    blocking_stderr: float
    first_half_mean: float
    second_half_mean: float
    first_second_half_z: float
    cumulative_mean_drift: float
    block_count: int
    block_means: tuple[float, ...]
    spread_warning: bool
    spread_veto: bool
    trend_clean: bool
    blocking_clean: bool
    cumulative_drift_clean: bool
    stationarity_clean: bool
def _finite_trace(times: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(times, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if times.size != values.size:
        raise ValueError("times and values must have the same length")
    mask = np.isfinite(times) & np.isfinite(values)
    return times[mask], values[mask]
def linear_slope_statistics(times: FloatArray, values: FloatArray) -> SlopeResult:
    times, values = _finite_trace(times, values)
    if times.size < 3:
        return SlopeResult(float("nan"), float("nan"), float("inf"), False)
    centered = times - float(np.mean(times))
    denom = float(np.sum(centered * centered))
    if denom <= 0.0:
        return SlopeResult(float("nan"), float("nan"), float("inf"), False)
    slope = float(np.sum(centered * (values - float(np.mean(values)))) / denom)
    intercept = float(np.mean(values) - slope * np.mean(times))
    residual = values - (intercept + slope * times)
    dof = times.size - 2
    sigma2 = float(np.sum(residual * residual) / dof)
    stderr = float(np.sqrt(sigma2 / denom))
    if stderr == 0.0:
        compatible = slope == 0.0
        z = 0.0 if compatible else float("inf")
    elif np.isfinite(stderr):
        z = abs(slope) / stderr
        compatible = z <= 2.0
    else:
        z = float("inf")
        compatible = False
    return SlopeResult(slope, stderr, float(z), bool(compatible))
def autocorrelation(values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.asarray([], dtype=float)
    centered = arr - float(np.mean(arr))
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        return np.ones(1, dtype=float)
    padded = np.zeros(2 * arr.size, dtype=float)
    padded[: arr.size] = centered
    spectrum = np.fft.rfft(padded)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=padded.size)[: arr.size]
    normalizer = variance * np.arange(arr.size, 0, -1, dtype=float) / arr.size
    return raw / normalizer
def integrated_autocorrelation_time(
    values: FloatArray,
    *,
    window_c: float = 5.0,
) -> AutocorrelationResult:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if _trace_is_constant(arr):
        return AutocorrelationResult(0.5, 0, float(arr.size), "zero_variance", np.ones(1))
    acf = autocorrelation(arr)
    if acf.size < 2:
        if acf.size == 1:
            return AutocorrelationResult(0.5, 0, float(arr.size), "zero_variance", acf)
        return AutocorrelationResult(float("nan"), None, float("nan"), "insufficient_points", acf)
    tau = 0.5
    window = acf.size - 1
    reason = "max_lag"
    for lag in range(1, acf.size):
        if acf[lag] <= 0.0:
            window = lag - 1
            reason = "first_nonpositive_acf"
            break
        tau += float(acf[lag])
        if lag >= window_c * tau:
            window = lag
            reason = "sokal_window"
            break
    tau = max(tau, 0.5)
    n_eff = float(arr.size / (2.0 * tau))
    return AutocorrelationResult(float(tau), int(window), n_eff, reason, acf)
def trace_stationarity_diagnostics(
    times: FloatArray,
    values: FloatArray,
    *,
    slope_z_threshold: float = 2.0,
    quarter_z_threshold: float = 2.5,
    cumulative_z_threshold: float = 2.0,
    spread_warning_z_threshold: float = 4.0,
) -> TraceStationarityResult:
    times, values = _finite_trace(times, values)
    if _trace_is_constant(values):
        return _constant_trace_stationarity(values)
    slope = linear_slope_statistics(times, values)
    acf = integrated_autocorrelation_time(values)
    n = int(values.size)
    if (
        n > 0
        and np.isfinite(acf.effective_independent_samples)
        and acf.effective_independent_samples > 0.0
    ):
        adjusted_stderr = slope.slope_stderr * np.sqrt(n / acf.effective_independent_samples)
        adjusted_z = _safe_abs_z(slope.slope, adjusted_stderr)
    else:
        adjusted_stderr = float("nan")
        adjusted_z = float("inf")
    cumulative = _cumulative_drift(values)
    first_last_quarter_z = cumulative.first_last_quarter_z
    late_cumulative_z = cumulative.late_cumulative_z
    block = _block_drift(values)
    trend_clean = bool(adjusted_z <= slope_z_threshold)
    cumulative_clean = bool(
        first_last_quarter_z <= quarter_z_threshold and late_cumulative_z <= cumulative_z_threshold
    )
    first_last_block_clean = bool(block.first_last_z <= quarter_z_threshold)
    spread_warning = bool(block.spread_z > spread_warning_z_threshold)
    spread_veto = bool(
        spread_warning and (not trend_clean or not cumulative_clean or not first_last_block_clean)
    )
    blocking_clean = bool(first_last_block_clean and not spread_veto)
    clean = bool(trend_clean and cumulative_clean and blocking_clean)
    return TraceStationarityResult(
        point_count=n,
        slope=slope,
        autocorrelation=acf,
        slope_stderr_autocorr_adjusted=float(adjusted_stderr),
        slope_z_autocorr_adjusted=float(adjusted_z),
        first_last_quarter_z=float(first_last_quarter_z),
        late_cumulative_z=float(late_cumulative_z),
        first_last_blocking_z=float(block.first_last_z),
        spread_blocking_z=float(block.spread_z),
        blocking_stderr=float(block.global_stderr),
        first_half_mean=float(cumulative.first_half_mean),
        second_half_mean=float(cumulative.second_half_mean),
        first_second_half_z=float(cumulative.first_second_half_z),
        cumulative_mean_drift=float(cumulative.cumulative_mean_drift),
        block_count=int(block.block_count),
        block_means=tuple(block.means),
        spread_warning=bool(spread_warning),
        spread_veto=bool(spread_veto),
        trend_clean=bool(trend_clean),
        blocking_clean=bool(blocking_clean),
        cumulative_drift_clean=bool(cumulative_clean),
        stationarity_clean=clean,
    )
def _blocking_stderr(values: FloatArray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    try:
        return float(blocking_standard_error(values).plateau_stderr)
    except ValueError:
        return float("nan")
@dataclass(frozen=True)
class _CumulativeDrift:
    first_last_quarter_z: float
    late_cumulative_z: float
    first_half_mean: float
    second_half_mean: float
    first_second_half_z: float
    cumulative_mean_drift: float
def _cumulative_drift(values: FloatArray) -> _CumulativeDrift:
    n = int(values.size)
    if n < 8:
        return _CumulativeDrift(
            first_last_quarter_z=float("inf"),
            late_cumulative_z=float("inf"),
            first_half_mean=float("nan"),
            second_half_mean=float("nan"),
            first_second_half_z=float("inf"),
            cumulative_mean_drift=float("nan"),
        )
    quarter = max(2, n // 4)
    first = values[:quarter]
    last = values[-quarter:]
    first_se = _blocking_stderr(first)
    last_se = _blocking_stderr(last)
    combined = float(np.sqrt(first_se**2 + last_se**2))
    quarter_z = _safe_abs_z(float(np.mean(last)) - float(np.mean(first)), combined)
    cumulative = np.cumsum(values) / np.arange(1, n + 1, dtype=float)
    anchor_index = max(0, int(0.75 * n) - 1)
    late_delta = float(cumulative[-1] - cumulative[anchor_index])
    global_se = _blocking_stderr(values)
    late_z = _safe_abs_z(late_delta, global_se)
    half = max(2, n // 2)
    first_half = values[:half]
    second_half = values[-half:]
    first_half_mean = float(np.mean(first_half))
    second_half_mean = float(np.mean(second_half))
    half_combined = float(
        np.sqrt(_blocking_stderr(first_half) ** 2 + _blocking_stderr(second_half) ** 2)
    )
    return _CumulativeDrift(
        first_last_quarter_z=float(quarter_z),
        late_cumulative_z=float(late_z),
        first_half_mean=first_half_mean,
        second_half_mean=second_half_mean,
        first_second_half_z=_safe_abs_z(second_half_mean - first_half_mean, half_combined),
        cumulative_mean_drift=late_delta,
    )
@dataclass(frozen=True)
class _BlockDrift:
    first_last_z: float
    spread_z: float
    global_stderr: float
    block_count: int
    means: tuple[float, ...]
def _block_drift(values: FloatArray, block_count: int = 4) -> _BlockDrift:
    if values.size < 2 * block_count:
        return _BlockDrift(float("inf"), float("inf"), float("nan"), block_count, ())
    chunks = np.array_split(values, block_count)
    means = np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=float)
    first_se = _blocking_stderr(chunks[0])
    last_se = _blocking_stderr(chunks[-1])
    combined = float(np.sqrt(first_se**2 + last_se**2))
    first_last_z = _safe_abs_z(float(means[-1] - means[0]), combined)
    stderr = _blocking_stderr(values)
    spread = float(np.max(means) - np.min(means))
    spread_z = _safe_abs_z(spread, stderr)
    return _BlockDrift(
        first_last_z=float(first_last_z),
        spread_z=float(spread_z),
        global_stderr=float(stderr),
        block_count=block_count,
        means=tuple(float(value) for value in means),
    )
def _safe_abs_z(delta: float, stderr: float) -> float:
    if stderr > 0.0 and np.isfinite(stderr):
        return float(abs(delta) / stderr)
    return 0.0 if delta == 0.0 else float("inf")
def _trace_is_constant(values: FloatArray) -> bool:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    scale = max(1.0, float(np.max(np.abs(arr))))
    return bool(float(np.max(arr) - np.min(arr)) <= 1.0e-12 * scale)
def _constant_trace_stationarity(values: FloatArray) -> TraceStationarityResult:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    block_count = 4
    means = tuple(mean for _ in range(block_count)) if n >= 2 * block_count else ()
    return TraceStationarityResult(
        point_count=n,
        slope=SlopeResult(0.0, 0.0, 0.0, True),
        autocorrelation=AutocorrelationResult(0.5, 0, float(n), "zero_variance", np.ones(1)),
        slope_stderr_autocorr_adjusted=0.0,
        slope_z_autocorr_adjusted=0.0,
        first_last_quarter_z=0.0,
        late_cumulative_z=0.0,
        first_last_blocking_z=0.0,
        spread_blocking_z=0.0,
        blocking_stderr=0.0,
        first_half_mean=mean,
        second_half_mean=mean,
        first_second_half_z=0.0,
        cumulative_mean_drift=0.0,
        block_count=block_count,
        block_means=means,
        spread_warning=False,
        spread_veto=False,
        trend_clean=True,
        blocking_clean=True,
        cumulative_drift_clean=True,
        stationarity_clean=True,
    )

FloatArray = NDArray[np.float64]
CHAIN_ACCEPTED = "accepted"
CHAIN_RHAT_ABOVE_LIMIT = "rhat_above_limit"
CHAIN_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM = "effective_sample_count_below_minimum"
CHAIN_TRACE_NONSTATIONARY = "trace_nonstationary"
CHAIN_SPREAD_WARNING = "spread_warning"
@dataclass(frozen=True)
class ChainObservableDiagnostics:
    rhat: float
    min_effective_independent_samples: float
    stationarity_clean_count: int
    spread_warning_count: int
    chain_count: int
    classification: str
    chain_diagnostics: list[TraceStationarityResult]
    def to_dict(self) -> dict:
        return {
            "rhat": self.rhat,
            "min_effective_independent_samples": self.min_effective_independent_samples,
            "stationarity_clean_count": self.stationarity_clean_count,
            "spread_warning_count": self.spread_warning_count,
            "chain_count": self.chain_count,
            "classification": self.classification,
            "chain_diagnostics": [
                trace_stationarity_result_to_dict(result) for result in self.chain_diagnostics
            ],
        }
def diagnose_chains(
    times_by_chain: list[FloatArray],
    values_by_chain: list[FloatArray],
    *,
    rhat_threshold: float = 1.05,
    min_effective_samples: float = 30.0,
) -> ChainObservableDiagnostics:
    if len(times_by_chain) != len(values_by_chain):
        raise ValueError("times_by_chain and values_by_chain must have the same length")
    if not values_by_chain:
        raise ValueError("at least one chain is required")
    diagnostics = [
        trace_stationarity_diagnostics(times, values)
        for times, values in zip(times_by_chain, values_by_chain, strict=True)
    ]
    rhat = split_rhat(values_by_chain)
    neff_values = [
        result.autocorrelation.effective_independent_samples
        for result in diagnostics
        if np.isfinite(result.autocorrelation.effective_independent_samples)
    ]
    min_neff = min(neff_values) if neff_values else float("nan")
    clean_count = sum(result.stationarity_clean for result in diagnostics)
    spread_count = sum(result.spread_warning for result in diagnostics)
    classification = _classify_chain_diagnostics(
        rhat=rhat,
        min_neff=min_neff,
        clean_count=clean_count,
        chain_count=len(diagnostics),
        spread_warning_count=spread_count,
        rhat_threshold=rhat_threshold,
        min_effective_samples=min_effective_samples,
    )
    return ChainObservableDiagnostics(
        rhat=float(rhat),
        min_effective_independent_samples=float(min_neff),
        stationarity_clean_count=int(clean_count),
        spread_warning_count=int(spread_count),
        chain_count=len(diagnostics),
        classification=classification,
        chain_diagnostics=diagnostics,
    )
def split_rhat(chains: list[FloatArray]) -> float:
    finite_chains = [_finite_1d(chain) for chain in chains]
    finite_chains = [chain for chain in finite_chains if chain.size >= 4]
    if len(finite_chains) < 2:
        return float("nan")
    min_size = min(chain.size for chain in finite_chains)
    half = min_size // 2
    if half < 2:
        return float("nan")
    split = []
    for chain in finite_chains:
        trimmed = chain[-min_size:]
        split.append(trimmed[:half])
        split.append(trimmed[half : 2 * half])
    values = np.vstack(split)
    _chain_count, draw_count = values.shape
    means = np.mean(values, axis=1)
    variances = np.var(values, axis=1, ddof=1)
    within = float(np.mean(variances))
    if within == 0.0:
        return 1.0 if float(np.var(means)) == 0.0 else float("inf")
    between = float(draw_count * np.var(means, ddof=1))
    variance_hat = ((draw_count - 1.0) / draw_count) * within + between / draw_count
    return float(np.sqrt(max(variance_hat / within, 0.0)))
def _classify_chain_diagnostics(
    *,
    rhat: float,
    min_neff: float,
    clean_count: int,
    chain_count: int,
    spread_warning_count: int,
    rhat_threshold: float = 1.05,
    min_effective_samples: float = 30.0,
) -> str:
    if not np.isfinite(rhat) or rhat >= rhat_threshold:
        return CHAIN_RHAT_ABOVE_LIMIT
    if not np.isfinite(min_neff) or min_neff < min_effective_samples:
        return CHAIN_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM
    if clean_count < chain_count:
        return CHAIN_TRACE_NONSTATIONARY
    if spread_warning_count > 0:
        return CHAIN_SPREAD_WARNING
    return CHAIN_ACCEPTED
def trace_stationarity_result_to_dict(result: TraceStationarityResult) -> dict:
    return {
        "point_count": result.point_count,
        "slope_z_autocorr_adjusted": result.slope_z_autocorr_adjusted,
        "first_last_quarter_z": result.first_last_quarter_z,
        "late_cumulative_z": result.late_cumulative_z,
        "first_last_blocking_z": result.first_last_blocking_z,
        "spread_blocking_z": result.spread_blocking_z,
        "blocking_stderr": result.blocking_stderr,
        "first_half_mean": result.first_half_mean,
        "second_half_mean": result.second_half_mean,
        "first_second_half_z": result.first_second_half_z,
        "cumulative_mean_drift": result.cumulative_mean_drift,
        "block_count": result.block_count,
        "block_means": list(result.block_means),
        "effective_independent_samples": result.autocorrelation.effective_independent_samples,
        "tau_int_samples": result.autocorrelation.tau_int_samples,
        "spread_warning": result.spread_warning,
        "spread_veto": result.spread_veto,
        "trend_clean": result.trend_clean,
        "blocking_clean": result.blocking_clean,
        "cumulative_drift_clean": result.cumulative_drift_clean,
        "stationarity_clean": result.stationarity_clean,
    }
def _finite_1d(values: FloatArray) -> FloatArray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]
