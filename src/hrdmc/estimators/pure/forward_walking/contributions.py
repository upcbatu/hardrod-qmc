from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ContributionEventLike(Protocol):
    @property
    def positions(self) -> FloatArray: ...


def raw_r2_contribution_per_walker(
    positions: FloatArray,
    *,
    center: float = 0.0,
) -> FloatArray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    return np.mean((values - center) ** 2, axis=1)


def rao_blackwell_r2_contribution_per_walker(
    positions: FloatArray,
    *,
    center: float,
    com_variance: float,
) -> FloatArray:
    """Return the COM Rao-Blackwellized R2 contribution for each walker."""

    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if not np.isfinite(com_variance) or com_variance < 0.0:
        raise ValueError("com_variance must be finite and non-negative")
    centered = values - center
    relative = centered - np.mean(centered, axis=1, keepdims=True)
    return np.mean(relative * relative, axis=1) + float(com_variance)


def event_r2_contributions(
    event: ContributionEventLike,
    *,
    source: str,
    center: float,
    r2_rb_com_variance: float | None,
) -> FloatArray:
    if source == "raw_r2":
        return raw_r2_contribution_per_walker(event.positions, center=center)
    if source == "r2_rb":
        if r2_rb_com_variance is None:
            raise ValueError("r2_rb requires an explicit COM variance")
        return rao_blackwell_r2_contribution_per_walker(
            event.positions,
            center=center,
            com_variance=r2_rb_com_variance,
        )
    raise ValueError(f"unsupported observable source: {source}")


def event_contribution_matrix(
    event: ContributionEventLike,
    *,
    observable: str,
    center: float,
    observable_source: str,
    r2_rb_com_variance: float | None = None,
    density_bin_edges: FloatArray | None = None,
    pair_bin_edges: FloatArray | None = None,
    structure_k_values: FloatArray | None = None,
) -> FloatArray:
    if observable == "r2":
        values = event_r2_contributions(
            event,
            source=observable_source,
            center=center,
            r2_rb_com_variance=r2_rb_com_variance,
        )
        return values[:, np.newaxis]
    if observable == "density":
        return density_profile_matrix(event.positions, bin_edges=density_bin_edges)
    if observable == "pair_distance_density":
        return pair_distance_density_matrix(event.positions, bin_edges=pair_bin_edges)
    if observable == "structure_factor":
        return structure_factor_matrix(event.positions, k_values=structure_k_values)
    raise ValueError(f"unsupported observable: {observable}")


def density_profile_matrix(positions: FloatArray, *, bin_edges: FloatArray | None) -> FloatArray:
    edges = _required_edges(bin_edges, "density_bin_edges")
    widths = np.diff(edges)
    samples = np.asarray(positions, dtype=float)
    if samples.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("positions must be finite")
    out = np.zeros((samples.shape[0], edges.size - 1), dtype=float)
    indices = np.searchsorted(edges, samples, side="right") - 1
    in_grid = (indices >= 0) & (indices < out.shape[1])
    walker_indices = np.broadcast_to(np.arange(samples.shape[0])[:, np.newaxis], indices.shape)
    np.add.at(out, (walker_indices[in_grid], indices[in_grid]), 1.0)
    out /= widths
    return out


def weighted_density_profile(
    positions: FloatArray,
    *,
    bin_edges: FloatArray | None,
    walker_weights: FloatArray,
) -> FloatArray:
    edges = _required_edges(bin_edges, "density_bin_edges")
    widths = np.diff(edges)
    samples = np.asarray(positions, dtype=float)
    weights = np.asarray(walker_weights, dtype=float)
    if samples.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if weights.shape != (samples.shape[0],):
        raise ValueError("walker_weights must have one value per walker")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)):
        raise ValueError("positions and walker_weights must be finite")
    out = np.zeros(edges.size - 1, dtype=float)
    indices = np.searchsorted(edges, samples, side="right") - 1
    in_grid = (indices >= 0) & (indices < out.size)
    weighted_particles = np.broadcast_to(weights[:, np.newaxis], samples.shape)
    np.add.at(out, indices[in_grid], weighted_particles[in_grid])
    out /= widths
    return out


def add_density_profile_to_auxiliary(
    auxiliary: FloatArray,
    positions: FloatArray,
    *,
    bin_edges: FloatArray | None,
) -> None:
    edges = _required_edges(bin_edges, "density_bin_edges")
    widths = np.diff(edges)
    samples = np.asarray(positions, dtype=float)
    if samples.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if auxiliary.shape != (samples.shape[0], edges.size - 1):
        raise ValueError("density auxiliary shape does not match positions and bins")
    if not np.all(np.isfinite(samples)):
        raise ValueError("positions must be finite")
    indices = np.searchsorted(edges, samples, side="right") - 1
    in_grid = (indices >= 0) & (indices < auxiliary.shape[1])
    walker_indices = np.broadcast_to(np.arange(samples.shape[0])[:, np.newaxis], indices.shape)
    np.add.at(
        auxiliary,
        (walker_indices[in_grid], indices[in_grid]),
        1.0 / widths[indices[in_grid]],
    )


def pair_distance_density_matrix(
    positions: FloatArray,
    *,
    bin_edges: FloatArray | None,
) -> FloatArray:
    edges = _required_edges(bin_edges, "pair_bin_edges")
    widths = np.diff(edges)
    samples = np.asarray(positions, dtype=float)
    if samples.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("positions must be finite")
    n_particles = samples.shape[1]
    out = np.zeros((samples.shape[0], edges.size - 1), dtype=float)
    i, j = _pair_indices(n_particles)
    distances = np.abs(samples[:, j] - samples[:, i])
    indices = np.searchsorted(edges, distances, side="right") - 1
    in_grid = (indices >= 0) & (indices < out.shape[1])
    walker_indices = np.broadcast_to(np.arange(samples.shape[0])[:, np.newaxis], indices.shape)
    np.add.at(out, (walker_indices[in_grid], indices[in_grid]), 1.0)
    out /= widths
    return out


def structure_factor_matrix(
    positions: FloatArray,
    *,
    k_values: FloatArray | None,
) -> FloatArray:
    k = np.asarray(k_values, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("structure_k_values must be provided for structure_factor")
    samples = np.asarray(positions, dtype=float)
    if samples.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("positions must be finite")
    phase = np.exp(1j * samples[:, np.newaxis, :] * k[np.newaxis, :, np.newaxis])
    rho_k = np.sum(phase, axis=2)
    return (np.abs(rho_k) ** 2) / float(samples.shape[1])


def _required_edges(edges: FloatArray | None, name: str) -> FloatArray:
    if edges is None:
        raise ValueError(f"{name} must be provided")
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be provided")
    return values


@lru_cache(maxsize=16)
def _pair_indices(n_particles: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return np.triu_indices(n_particles, k=1)
