from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RunningStats:
    """Mergeable Welford mean/variance accumulator."""

    count: int
    mean: float
    m2: float

    @classmethod
    def empty(cls) -> RunningStats:
        return cls(count=0, mean=0.0, m2=0.0)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("nan")
        return self.m2 / (self.count - 1)

    @property
    def standard_error(self) -> float:
        if self.count < 2:
            return float("nan")
        return float(np.sqrt(self.variance / self.count))

    def update(self, value: float) -> RunningStats:
        if not np.isfinite(value):
            return self
        next_count = self.count + 1
        delta = value - self.mean
        next_mean = self.mean + delta / next_count
        next_m2 = self.m2 + delta * (value - next_mean)
        return RunningStats(count=next_count, mean=float(next_mean), m2=float(next_m2))

    def update_many(self, values: FloatArray) -> RunningStats:
        out = self
        for value in np.asarray(values, dtype=float).reshape(-1):
            out = out.update(float(value))
        return out

    def merge(self, other: RunningStats) -> RunningStats:
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / total
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        return RunningStats(count=total, mean=float(mean), m2=float(m2))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "standard_error": self.standard_error,
            "m2": self.m2,
        }


@dataclass(frozen=True)
class RunningHistogram:
    """Mergeable weighted histogram accumulator with out-of-grid accounting."""

    bin_edges: FloatArray
    counts: FloatArray
    sample_count: int
    total_weight: float
    lost_sample_count: int
    lost_weight: float

    @classmethod
    def from_edges(cls, bin_edges: FloatArray) -> RunningHistogram:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional array with at least two edges")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("bin_edges must be strictly increasing")
        return cls(
            bin_edges=edges,
            counts=np.zeros(edges.size - 1, dtype=float),
            sample_count=0,
            total_weight=0.0,
            lost_sample_count=0,
            lost_weight=0.0,
        )

    @classmethod
    def from_centers(cls, grid: FloatArray) -> RunningHistogram:
        centers = np.asarray(grid, dtype=float)
        if centers.ndim != 1 or centers.size < 2:
            raise ValueError("grid must be a one-dimensional array with at least two centers")
        dx = float(centers[1] - centers[0])
        if dx <= 0.0 or not np.allclose(np.diff(centers), dx):
            raise ValueError("grid must be uniformly increasing")
        edges = np.concatenate(
            ([centers[0] - 0.5 * dx], 0.5 * (centers[:-1] + centers[1:]), [centers[-1] + 0.5 * dx])
        )
        return cls.from_edges(edges)

    @property
    def bin_widths(self) -> FloatArray:
        return np.diff(self.bin_edges)

    @property
    def density(self) -> FloatArray:
        return self.counts / self.bin_widths

    @property
    def density_integral(self) -> float:
        return float(np.sum(self.density * self.bin_widths))

    def update(self, values: FloatArray, weights: FloatArray | None = None) -> RunningHistogram:
        values = np.asarray(values, dtype=float).reshape(-1)
        if weights is None:
            weights = np.ones(values.size, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape != values.shape:
            raise ValueError("weights must match values")

        finite = np.isfinite(values) & np.isfinite(weights)
        values = values[finite]
        weights = weights[finite]
        if values.size == 0:
            return self

        indices = np.searchsorted(self.bin_edges, values, side="right") - 1
        in_grid = (indices >= 0) & (indices < self.counts.size)
        next_counts = self.counts.copy()
        next_counts += np.bincount(
            indices[in_grid],
            weights=weights[in_grid],
            minlength=self.counts.size,
        )
        return RunningHistogram(
            bin_edges=self.bin_edges,
            counts=next_counts,
            sample_count=self.sample_count + int(values.size),
            total_weight=self.total_weight + float(np.sum(weights)),
            lost_sample_count=self.lost_sample_count + int(np.count_nonzero(~in_grid)),
            lost_weight=self.lost_weight + float(np.sum(weights[~in_grid])),
        )

    def merge(self, other: RunningHistogram) -> RunningHistogram:
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise ValueError("histogram bin_edges must match")
        return RunningHistogram(
            bin_edges=self.bin_edges,
            counts=self.counts + other.counts,
            sample_count=self.sample_count + other.sample_count,
            total_weight=self.total_weight + other.total_weight,
            lost_sample_count=self.lost_sample_count + other.lost_sample_count,
            lost_weight=self.lost_weight + other.lost_weight,
        )

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "counts": self.counts.tolist(),
            "sample_count": self.sample_count,
            "total_weight": self.total_weight,
            "density_integral": self.density_integral,
            "lost_sample_count": self.lost_sample_count,
            "lost_weight": self.lost_weight,
        }
