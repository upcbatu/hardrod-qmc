from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

FloatArray = NDArray[np.float64]

RANK_DIAGNOSTICS_OK = "ok"
RANK_DIAGNOSTICS_CONSTANT = "constant"
RANK_DIAGNOSTICS_BETWEEN_CHAIN_DISAGREEMENT = "between_chain_disagreement"
RANK_DIAGNOSTICS_INSUFFICIENT_INFORMATION = "insufficient_information"


@dataclass(frozen=True)
class RankNormalizedDiagnostics:
    status: str
    chain_count: int
    draws_per_chain: int
    split_chain_count: int
    split_draws_per_chain: int
    rank_split_rhat: float
    folded_split_rhat: float
    split_rhat: float
    bulk_ess: float
    bulk_ess_per_chain: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "chain_count": self.chain_count,
            "draws_per_chain": self.draws_per_chain,
            "split_chain_count": self.split_chain_count,
            "split_draws_per_chain": self.split_draws_per_chain,
            "rank_split_rhat": self.rank_split_rhat,
            "folded_split_rhat": self.folded_split_rhat,
            "split_rhat": self.split_rhat,
            "bulk_ess": self.bulk_ess,
            "bulk_ess_per_chain": list(self.bulk_ess_per_chain),
        }


def rank_normalized_diagnostics(chains: FloatArray) -> RankNormalizedDiagnostics:
    """Return rank-normalized split-Rhat and bulk ESS following Vehtari [Vehtari2021]."""
    values = np.asarray(chains, dtype=float)
    if values.ndim != 2:
        raise ValueError("chains must be a two-dimensional rectangular array")
    if not np.all(np.isfinite(values)):
        raise ValueError("chains must contain only finite values")
    chain_count, draw_count = values.shape
    even_draw_count = draw_count - draw_count % 2
    split_draw_count = even_draw_count // 2
    if chain_count < 2 or draw_count < 4:
        return _empty_result(
            status=RANK_DIAGNOSTICS_INSUFFICIENT_INFORMATION,
            chain_count=chain_count,
            draw_count=draw_count,
        )
    if np.all(values == values[0, 0]):
        total = float(chain_count * draw_count)
        return RankNormalizedDiagnostics(
            status=RANK_DIAGNOSTICS_CONSTANT,
            chain_count=chain_count,
            draws_per_chain=draw_count,
            split_chain_count=2 * chain_count,
            split_draws_per_chain=split_draw_count,
            rank_split_rhat=1.0,
            folded_split_rhat=1.0,
            split_rhat=1.0,
            bulk_ess=total,
            bulk_ess_per_chain=tuple(float(draw_count) for _ in range(chain_count)),
        )
    constant_by_chain = np.all(values == values[:, :1], axis=1)
    if np.all(constant_by_chain):
        trimmed = values[:, :even_draw_count]
        split = np.concatenate(
            (trimmed[:, :split_draw_count], trimmed[:, split_draw_count:]), axis=0
        )
        folded_rhat = _split_rhat_from_transformed(
            _rank_normalize(np.abs(split - np.median(split)))
        )
        return RankNormalizedDiagnostics(
            status=RANK_DIAGNOSTICS_BETWEEN_CHAIN_DISAGREEMENT,
            chain_count=chain_count,
            draws_per_chain=draw_count,
            split_chain_count=2 * chain_count,
            split_draws_per_chain=split_draw_count,
            rank_split_rhat=float("inf"),
            folded_split_rhat=folded_rhat,
            split_rhat=float("inf"),
            bulk_ess=0.0,
            bulk_ess_per_chain=tuple(float(draw_count) for _ in range(chain_count)),
        )
    trimmed = values[:, :even_draw_count]
    split = np.concatenate(
        (trimmed[:, :split_draw_count], trimmed[:, split_draw_count:]), axis=0
    )
    rank_values = _rank_normalize(split)
    folded_values = _rank_normalize(np.abs(split - np.median(split)))
    rank_rhat = _split_rhat_from_transformed(rank_values)
    folded_rhat = _split_rhat_from_transformed(folded_values)
    split_rhat = max(rank_rhat, folded_rhat)
    bulk_ess = _geyer_bulk_ess(rank_values)
    unsplit_rank_values = _rank_normalize(values)
    per_chain = tuple(
        float(_geyer_bulk_ess(unsplit_rank_values[index : index + 1, :]))
        for index in range(chain_count)
    )
    return RankNormalizedDiagnostics(
        status=RANK_DIAGNOSTICS_OK,
        chain_count=chain_count,
        draws_per_chain=draw_count,
        split_chain_count=2 * chain_count,
        split_draws_per_chain=split_draw_count,
        rank_split_rhat=float(rank_rhat),
        folded_split_rhat=float(folded_rhat),
        split_rhat=float(split_rhat),
        bulk_ess=float(bulk_ess),
        bulk_ess_per_chain=per_chain,
    )


def _empty_result(*, status: str, chain_count: int, draw_count: int) -> RankNormalizedDiagnostics:
    return RankNormalizedDiagnostics(
        status=status,
        chain_count=chain_count,
        draws_per_chain=draw_count,
        split_chain_count=0,
        split_draws_per_chain=0,
        rank_split_rhat=float("nan"),
        folded_split_rhat=float("nan"),
        split_rhat=float("nan"),
        bulk_ess=float("nan"),
        bulk_ess_per_chain=tuple(float("nan") for _ in range(chain_count)),
    )


def _rank_normalize(values: FloatArray) -> FloatArray:
    ranks = rankdata(values.reshape(-1), method="average")
    sample_count = ranks.size
    probabilities = (ranks - 0.375) / (sample_count + 0.25)
    return np.asarray(norm.ppf(probabilities), dtype=float).reshape(values.shape)


def _split_rhat_from_transformed(values: FloatArray) -> float:
    chain_count, draw_count = values.shape
    chain_means = np.mean(values, axis=1)
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    if within == 0.0:
        return 1.0 if float(np.var(chain_means)) == 0.0 else float("inf")
    between = float(draw_count * np.var(chain_means, ddof=1))
    variance_hat = ((draw_count - 1.0) / draw_count) * within + between / draw_count
    if chain_count < 2:
        return float("nan")
    return float(np.sqrt(max(variance_hat / within, 0.0)))


def _geyer_bulk_ess(values: FloatArray) -> float:
    chain_count, draw_count = values.shape
    if draw_count < 3:
        return float("nan")
    if np.all(values == values[0, 0]):
        return float(chain_count * draw_count)
    autocovariances = np.vstack([_autocovariance(chain) for chain in values])
    within = float(np.mean(autocovariances[:, 0]) * draw_count / (draw_count - 1.0))
    variance_plus = within * (draw_count - 1.0) / draw_count
    if chain_count > 1:
        variance_plus += float(np.var(np.mean(values, axis=1), ddof=1))
    if not np.isfinite(variance_plus) or variance_plus <= 0.0:
        return 0.0
    rho = np.empty(draw_count, dtype=float)
    rho[0] = 1.0
    mean_autocovariance = np.mean(autocovariances, axis=0)
    rho[1:] = 1.0 - (within - mean_autocovariance[1:]) / variance_plus
    positive_pairs: list[float] = []
    for start in range(0, draw_count - 1, 2):
        pair_sum = float(rho[start] + rho[start + 1])
        if pair_sum <= 0.0:
            break
        if positive_pairs:
            pair_sum = min(pair_sum, positive_pairs[-1])
        positive_pairs.append(pair_sum)
    if not positive_pairs:
        return 0.0
    tau = -1.0 + 2.0 * float(np.sum(positive_pairs))
    total_draws = chain_count * draw_count
    minimum_tau = 1.0 / np.log10(max(total_draws, 10))
    tau = max(tau, minimum_tau)
    return float(total_draws / tau)


def _autocovariance(values: FloatArray) -> FloatArray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    draw_count = centered.size
    transform = np.fft.rfft(centered, n=2 * draw_count)
    covariance = np.fft.irfft(transform * np.conjugate(transform), n=2 * draw_count)[:draw_count]
    return np.asarray(covariance / draw_count, dtype=float)
