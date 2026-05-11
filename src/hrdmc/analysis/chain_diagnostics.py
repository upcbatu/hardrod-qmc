from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.analysis.timeseries import TraceStationarityResult, trace_stationarity_diagnostics

FloatArray = NDArray[np.float64]


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
    classification = classify_chain_diagnostics(
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
    chain_count, draw_count = values.shape
    means = np.mean(values, axis=1)
    variances = np.var(values, axis=1, ddof=1)
    within = float(np.mean(variances))
    if within == 0.0:
        return 1.0 if float(np.var(means)) == 0.0 else float("inf")
    between = float(draw_count * np.var(means, ddof=1))
    variance_hat = ((draw_count - 1.0) / draw_count) * within + between / draw_count
    return float(np.sqrt(max(variance_hat / within, 0.0)))


def classify_chain_diagnostics(
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
        return "NO_GO_RHAT"
    if not np.isfinite(min_neff) or min_neff < min_effective_samples:
        return "NO_GO_NEFF"
    if clean_count < chain_count:
        return "NO_GO_STATIONARITY"
    if spread_warning_count > 0:
        return "WARNING_SPREAD_ONLY"
    return "GO"


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
