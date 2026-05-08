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
