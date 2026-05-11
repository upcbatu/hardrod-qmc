from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BlockingResult:
    block_sizes: FloatArray
    stderr: FloatArray
    n_blocks: FloatArray

    @property
    def plateau_stderr(self) -> float:
        if len(self.stderr) == 0:
            return float("nan")
        return float(self.stderr[-1])


@dataclass(frozen=True)
class BlockingPlateauResult:
    plateau_found: bool
    plateau_stderr: float
    plateau_block_size: int
    plateau_n_blocks: int
    reason: str


def blocking_standard_error(series: FloatArray, min_blocks: int = 8) -> BlockingResult:
    """Estimate standard error using blocking analysis.

    Source/rationale
    ----------------
    Blocking is included because VMC/DMC samples are correlated in Markov-chain
    time. The method follows the blocking-error idea of Flyvbjerg and Petersen,
    J. Chem. Phys. 91, 461 (1989). [FlyvbjergPetersen1989Blocking]

    The function repeatedly doubles block size and computes the standard error
    of block means. It returns the sequence; users should inspect plateau
    behavior rather than blindly trusting a single number.
    """
    x = np.asarray(series, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        raise ValueError("series must contain at least two samples")
    if min_blocks < 2:
        raise ValueError("min_blocks must be at least 2")

    block_sizes: list[int] = []
    errors: list[float] = []
    n_blocks_list: list[int] = []

    block = 1
    while n // block >= min_blocks:
        nb = n // block
        trimmed = x[: nb * block]
        means = trimmed.reshape(nb, block).mean(axis=1)
        err = float(np.std(means, ddof=1) / np.sqrt(nb)) if nb > 1 else float("nan")
        block_sizes.append(block)
        errors.append(err)
        n_blocks_list.append(nb)
        block *= 2

    return BlockingResult(
        block_sizes=np.asarray(block_sizes, dtype=float),
        stderr=np.asarray(errors, dtype=float),
        n_blocks=np.asarray(n_blocks_list, dtype=float),
    )


def blocking_curve(series: FloatArray, min_blocks: int = 16) -> BlockingResult:
    """Return the blocking-error curve without selecting a point blindly."""

    return blocking_standard_error(series, min_blocks=min_blocks)


def detect_blocking_plateau(
    block_sizes: FloatArray,
    n_blocks: FloatArray,
    stderr: FloatArray,
    *,
    min_blocks: int = 16,
    window: int = 3,
    rel_tol: float = 0.10,
) -> BlockingPlateauResult:
    """Detect a stable tail in a blocking standard-error curve.

    The detector only considers curve points with enough remaining blocks. It
    does not blindly select the final point when the blocked sample count is
    too small.
    """

    if min_blocks < 2:
        raise ValueError("min_blocks must be at least 2")
    if window < 2:
        raise ValueError("window must be at least 2")
    if rel_tol < 0.0:
        raise ValueError("rel_tol must be non-negative")

    sizes = np.asarray(block_sizes, dtype=float).reshape(-1)
    counts = np.asarray(n_blocks, dtype=float).reshape(-1)
    errors = np.asarray(stderr, dtype=float).reshape(-1)
    if sizes.shape != counts.shape or sizes.shape != errors.shape:
        raise ValueError("block_sizes, n_blocks, and stderr must have matching shapes")

    valid = (
        np.isfinite(sizes)
        & np.isfinite(counts)
        & np.isfinite(errors)
        & (counts >= min_blocks)
        & (errors >= 0.0)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < window:
        return BlockingPlateauResult(
            plateau_found=False,
            plateau_stderr=float("nan"),
            plateau_block_size=0,
            plateau_n_blocks=0,
            reason="NO_GO_NO_BLOCKING_PLATEAU",
        )

    tail_indices = valid_indices[-window:]
    tail_errors = errors[tail_indices]
    scale = max(float(abs(np.median(tail_errors))), float(np.max(np.abs(tail_errors))), 1e-300)
    relative_spread = float((np.max(tail_errors) - np.min(tail_errors)) / scale)
    if relative_spread > rel_tol:
        return BlockingPlateauResult(
            plateau_found=False,
            plateau_stderr=float("nan"),
            plateau_block_size=int(sizes[tail_indices[-1]]),
            plateau_n_blocks=int(counts[tail_indices[-1]]),
            reason="NO_GO_NO_BLOCKING_PLATEAU",
        )

    selected = int(tail_indices[-1])
    return BlockingPlateauResult(
        plateau_found=True,
        plateau_stderr=float(np.max(tail_errors)),
        plateau_block_size=int(sizes[selected]),
        plateau_n_blocks=int(counts[selected]),
        reason="PLATEAU_FOUND",
    )
