from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ContributionEventLike(Protocol):
    @property
    def positions(self) -> FloatArray: ...

    @property
    def r2_rb_per_walker(self) -> FloatArray | None: ...


def raw_r2_contribution_per_walker(
    positions: FloatArray,
    *,
    center: float = 0.0,
) -> FloatArray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2:
        raise ValueError("positions must have shape (walkers, particles)")
    return np.mean((values - center) ** 2, axis=1)


def event_r2_contributions(
    event: ContributionEventLike,
    *,
    source: str,
    center: float,
) -> FloatArray:
    if source == "raw_r2":
        return raw_r2_contribution_per_walker(event.positions, center=center)
    if source == "r2_rb":
        if event.r2_rb_per_walker is None:
            raise ValueError("event does not carry r2_rb_per_walker")
        values = np.asarray(event.r2_rb_per_walker, dtype=float)
        if values.shape != (event.positions.shape[0],):
            raise ValueError("r2_rb_per_walker must have one value per walker")
        return values
    raise ValueError(f"unsupported observable source: {source}")


def event_contribution_matrix(
    event: ContributionEventLike,
    *,
    observable: str,
    center: float,
    observable_source: str,
    density_bin_edges: FloatArray | None = None,
    pair_bin_edges: FloatArray | None = None,
    structure_k_values: FloatArray | None = None,
) -> FloatArray:
    if observable == "r2":
        values = event_r2_contributions(
            event,
            source=observable_source,
            center=center,
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
    out = np.zeros((samples.shape[0], edges.size - 1), dtype=float)
    for walker, row in enumerate(samples):
        out[walker] = np.histogram(row, bins=edges)[0].astype(float) / widths
    return out


def pair_distance_density_matrix(
    positions: FloatArray,
    *,
    bin_edges: FloatArray | None,
) -> FloatArray:
    edges = _required_edges(bin_edges, "pair_bin_edges")
    widths = np.diff(edges)
    samples = np.asarray(positions, dtype=float)
    n_particles = samples.shape[1]
    out = np.zeros((samples.shape[0], edges.size - 1), dtype=float)
    for walker, row in enumerate(samples):
        distances = []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                distances.append(abs(float(row[j] - row[i])))
        out[walker] = (
            np.histogram(np.asarray(distances, dtype=float), bins=edges)[0].astype(float)
            / widths
        )
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
