from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _TruncatedGradientEstimate:
    """One unconditional hard-core cutoff estimate at ensemble or block level."""

    epsilon: float
    unconditional_t_grad: float
    excluded_probability: float


@dataclass(frozen=True)
class VariationalEnsembleMeans:
    """One ensemble-time reduction of variational oscillator-unit observables."""

    t_local: float
    trap: float
    e_local: float
    r2: float
    weighted_free_gap: float
    truncated_gradient: tuple[_TruncatedGradientEstimate, ...]

    @property
    def rms_radius(self) -> float:
        return float(np.sqrt(self.r2))


@dataclass(frozen=True)
class _VariationalBatchObservables:
    """Per-walker variational observables from one batched guide evaluation."""

    t_local: FloatArray
    trap: FloatArray
    e_local: FloatArray
    r2: FloatArray
    weighted_free_gap: FloatArray
    free_gaps: FloatArray
    cutoff_epsilons: FloatArray
    truncated_t_grad: FloatArray
    cutoff_excluded: NDArray[np.bool_]

    def __post_init__(self) -> None:
        fields = (
            "t_local",
            "trap",
            "e_local",
            "r2",
            "weighted_free_gap",
        )
        arrays = {name: _readonly_float_array(getattr(self, name)) for name in fields}
        walker_count = arrays["t_local"].size
        if any(array.shape != (walker_count,) for array in arrays.values()):
            raise ValueError("per-walker observable arrays must have one common shape")
        gaps = _readonly_float_array(self.free_gaps)
        if gaps.ndim != 2 or gaps.shape[0] != walker_count:
            raise ValueError("free_gaps must have shape (n_walkers, n_particles - 1)")
        cutoffs = _readonly_float_array(self.cutoff_epsilons)
        truncated = _readonly_float_array(self.truncated_t_grad)
        excluded = np.array(self.cutoff_excluded, dtype=bool, copy=True)
        if cutoffs.ndim != 1 or cutoffs.size < 1:
            raise ValueError("cutoff_epsilons must be a non-empty one-dimensional array")
        expected_cutoff_shape = (walker_count, cutoffs.size)
        if truncated.shape != expected_cutoff_shape or excluded.shape != expected_cutoff_shape:
            raise ValueError("cutoff arrays must have shape (n_walkers, n_cutoffs)")
        excluded.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "free_gaps", gaps)
        object.__setattr__(self, "cutoff_epsilons", cutoffs)
        object.__setattr__(self, "truncated_t_grad", truncated)
        object.__setattr__(self, "cutoff_excluded", excluded)

    @property
    def walker_count(self) -> int:
        return int(self.t_local.size)

    @property
    def particle_count(self) -> int:
        return int(self.free_gaps.shape[1] + 1)

    def ensemble_means(self) -> VariationalEnsembleMeans:
        """Average walkers once, preserving this batch as one time observation."""
        truncated = tuple(
            _TruncatedGradientEstimate(
                epsilon=float(epsilon),
                unconditional_t_grad=float(np.mean(self.truncated_t_grad[:, index])),
                excluded_probability=float(np.mean(self.cutoff_excluded[:, index])),
            )
            for index, epsilon in enumerate(self.cutoff_epsilons)
        )
        return VariationalEnsembleMeans(
            t_local=float(np.mean(self.t_local)),
            trap=float(np.mean(self.trap)),
            e_local=float(np.mean(self.e_local)),
            r2=float(np.mean(self.r2)),
            weighted_free_gap=float(np.mean(self.weighted_free_gap)),
            truncated_gradient=truncated,
        )


def _compose_variational_observables(
    positions: FloatArray,
    gradients: FloatArray,
    local_energies: FloatArray,
    *,
    valid_mask: NDArray[np.bool_],
    center: float,
    rod_length: float,
    cutoff_epsilons: FloatArray,
) -> _VariationalBatchObservables:
    """Compose observables from one precomputed, finite guide-state batch."""
    values = _positions_or_raise(positions)
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(rod_length) or rod_length < 0.0:
        raise ValueError("rod_length must be finite and non-negative")
    cutoffs = _validated_cutoffs(cutoff_epsilons)
    valid = _boolean_mask_or_raise(
        valid_mask,
        expected_shape=(values.shape[0],),
        name="precomputed valid mask",
    )
    if not np.all(valid):
        raise ValueError("precomputed valid mask rejected one or more walkers")
    gradient = np.asarray(gradients, dtype=float)
    local_energy = np.asarray(local_energies, dtype=float)
    if gradient.shape != values.shape:
        raise ValueError("guide gradient must match the position batch shape")
    if local_energy.shape != (values.shape[0],):
        raise ValueError("guide local energy must have shape (n_walkers,)")
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(local_energy))):
        raise ValueError("guide derivatives and local energies must be finite")
    free_gaps = np.diff(values, axis=1) - float(rod_length)
    if not np.all(np.isfinite(free_gaps)) or np.any(free_gaps < 0.0):
        raise ValueError("positions must satisfy the ordered hard-rod domain")
    gradient_squared = gradient * gradient
    t_grad = 0.5 * np.sum(gradient_squared, axis=1)
    centered_squared = (values - float(center)) ** 2
    trap = 0.5 * np.sum(centered_squared, axis=1)
    e_local = local_energy
    t_local = e_local - trap
    r2 = np.mean(centered_squared, axis=1)
    coefficients = _normalized_trap_gap_coefficients(values.shape[1])
    weighted_free_gap = free_gaps @ coefficients
    min_free_gap = np.min(free_gaps, axis=1)
    cutoff_excluded = min_free_gap[:, np.newaxis] < cutoffs[np.newaxis, :]
    truncated_t_grad = np.where(cutoff_excluded, 0.0, t_grad[:, np.newaxis])
    outputs = (
        t_local,
        trap,
        e_local,
        r2,
        weighted_free_gap,
        truncated_t_grad,
    )
    if not all(np.all(np.isfinite(output)) for output in outputs):
        raise ValueError("composed variational observables must be finite")
    return _VariationalBatchObservables(
        t_local=t_local,
        trap=trap,
        e_local=e_local,
        r2=r2,
        weighted_free_gap=weighted_free_gap,
        free_gaps=free_gaps,
        cutoff_epsilons=cutoffs,
        truncated_t_grad=truncated_t_grad,
        cutoff_excluded=cutoff_excluded,
    )


def _positions_or_raise(positions: FloatArray) -> FloatArray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("positions must have shape (n_walkers, n_particles) with both positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("positions must be finite")
    return values


def _normalized_trap_gap_coefficients(n_particles: int) -> FloatArray:
    k = np.arange(1, n_particles, dtype=float)
    coefficients = k * (n_particles - k)
    return coefficients / float(np.sum(coefficients))


def _boolean_mask_or_raise(
    values: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> NDArray[np.bool_]:
    mask = np.asarray(values)
    if mask.dtype != np.dtype(bool) or mask.shape != expected_shape:
        raise ValueError(f"{name} must be a boolean array with shape {expected_shape}")
    return mask


def _readonly_float_array(values: object) -> FloatArray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


if TYPE_CHECKING:
    from hrdmc.sampling.vmc.results import VMCTransitionEvent
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _VariationalHistogramRecord:
    """Immutable seed-aggregate histogram with explicit out-of-grid accounting."""

    bin_edges: tuple[float, ...]
    counts: tuple[int, ...]
    normalization_denominator: int
    out_of_grid_count: int
    expected_total_mass: float

    @property
    def in_grid_count(self) -> int:
        return int(sum(self.counts))

    @property
    def in_grid_mass(self) -> float:
        return float(self.in_grid_count / self.normalization_denominator)

    @property
    def out_of_grid_mass(self) -> float:
        return float(self.out_of_grid_count / self.normalization_denominator)

    @property
    def density(self) -> tuple[float, ...]:
        widths = np.diff(np.asarray(self.bin_edges, dtype=float))
        values = np.asarray(self.counts, dtype=float) / self.normalization_denominator / widths
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class _VariationalBlockRecord:
    """One compact time-ordered block of ensemble scalar means."""

    seed: int
    first_step: int
    last_step: int
    batch_count: int
    configuration_count: int
    particle_count: int
    means: VariationalEnsembleMeans


@dataclass(frozen=True)
class VariationalStreamResult:
    """Bounded, seed-preserving variational stream output."""

    seed: int
    block_size: int
    maximum_records: int
    production_transition_count: int
    configuration_count: int
    particle_count: int
    records: tuple[_VariationalBlockRecord, ...]
    density: _VariationalHistogramRecord
    free_gap_distribution: _VariationalHistogramRecord


class VariationalStreamingAccumulator:
    """Accumulate ensemble-time means without retaining walker histories."""

    _SCALAR_COUNT = 5

    def __init__(
        self,
        *,
        seed: int,
        block_size: int,
        maximum_records: int,
        density_bin_edges: FloatArray,
        free_gap_bin_edges: FloatArray,
        cutoff_epsilons: FloatArray,
    ) -> None:
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if maximum_records < 1:
            raise ValueError("maximum_records must be positive")
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.maximum_records = int(maximum_records)
        self._density_edges = _validated_edges(density_bin_edges, "density_bin_edges")
        self._free_gap_edges = _validated_edges(free_gap_bin_edges, "free_gap_bin_edges")
        self._cutoff_epsilons = _validated_cutoffs(cutoff_epsilons)
        self._records: list[_VariationalBlockRecord] = []
        self._finished_result: VariationalStreamResult | None = None
        self._last_step: int | None = None
        self._first_step: int | None = None
        self._batch_count = 0
        self._configuration_count = 0
        self._total_batch_count = 0
        self._total_configuration_count = 0
        self._particle_count: int | None = None
        self._scalar_sums = np.zeros(self._SCALAR_COUNT, dtype=float)
        self._truncated_t_grad_sums = np.zeros(self._cutoff_epsilons.size, dtype=float)
        self._excluded_probability_sums = np.zeros(self._cutoff_epsilons.size, dtype=float)
        self._density_counts = np.zeros(self._density_edges.size - 1, dtype=np.int64)
        self._density_out_of_grid = 0
        self._gap_counts = np.zeros(self._free_gap_edges.size - 1, dtype=np.int64)
        self._gap_out_of_grid = 0

    @property
    def cutoff_epsilons(self) -> FloatArray:
        return self._cutoff_epsilons

    def observe(
        self,
        *,
        step: int,
        positions: FloatArray,
        observables: _VariationalBatchObservables,
    ) -> None:
        """Consume one ensemble batch and discard its walker-level values."""
        values, step = self._validated_batch(step, positions, observables)
        means = observables.ensemble_means()
        scalar_values = np.asarray(
            [means.t_local, means.trap, means.e_local, means.r2, means.weighted_free_gap],
            dtype=float,
        )
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("ensemble means must be finite")
        self._accumulate(step, values, observables, means, scalar_values)
        if self._batch_count == self.block_size:
            self._seal_block()

    def _validated_batch(
        self,
        step: int,
        positions: FloatArray,
        observables: _VariationalBatchObservables,
    ) -> tuple[FloatArray, int]:
        if self._finished_result is not None:
            raise RuntimeError("cannot observe after finish")
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        step = int(step)
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("steps must be strictly increasing within one seed")
        values = np.asarray(positions, dtype=float)
        if values.shape != (observables.walker_count, observables.particle_count):
            raise ValueError("positions must match the variational observable batch")
        if not np.all(np.isfinite(values)):
            raise ValueError("positions must be finite")
        if self._particle_count is not None and observables.particle_count != self._particle_count:
            raise ValueError("particle_count cannot change within one accumulator")
        if not np.all(np.isfinite(observables.free_gaps)) or np.any(observables.free_gaps < 0.0):
            raise ValueError("free gaps must be finite and non-negative")
        if not np.array_equal(observables.cutoff_epsilons, self._cutoff_epsilons):
            raise ValueError("cutoff_epsilons cannot change within one accumulator")
        return values, step

    def _accumulate(
        self,
        step: int,
        values: FloatArray,
        observables: _VariationalBatchObservables,
        means: VariationalEnsembleMeans,
        scalar_values: FloatArray,
    ) -> None:
        if self._first_step is None:
            self._first_step = step
        self._last_step = step
        self._particle_count = observables.particle_count
        self._batch_count += 1
        self._configuration_count += observables.walker_count
        self._total_batch_count += 1
        self._total_configuration_count += observables.walker_count
        self._scalar_sums += scalar_values
        self._truncated_t_grad_sums += np.asarray(
            [estimate.unconditional_t_grad for estimate in means.truncated_gradient],
            dtype=float,
        )
        self._excluded_probability_sums += np.asarray(
            [estimate.excluded_probability for estimate in means.truncated_gradient],
            dtype=float,
        )
        density_counts, density_out = _histogram_counts(values.reshape(-1), self._density_edges)
        gap_counts, gap_out = _histogram_counts(
            observables.free_gaps.reshape(-1), self._free_gap_edges
        )
        self._density_counts += density_counts
        self._density_out_of_grid += density_out
        self._gap_counts += gap_counts
        self._gap_out_of_grid += gap_out

    def finish(self) -> VariationalStreamResult:
        """Seal a final partial block and return immutable seed-level records."""
        if self._finished_result is not None:
            return self._finished_result
        if self._total_batch_count < 1 or self._particle_count is None:
            raise RuntimeError("cannot finish an empty variational stream")
        if self._batch_count:
            self._seal_block()
        density = _VariationalHistogramRecord(
            bin_edges=tuple(float(value) for value in self._density_edges),
            counts=tuple(int(value) for value in self._density_counts),
            normalization_denominator=self._total_configuration_count,
            out_of_grid_count=self._density_out_of_grid,
            expected_total_mass=float(self._particle_count),
        )
        total_gap_count = self._total_configuration_count * (self._particle_count - 1)
        free_gap_distribution = _VariationalHistogramRecord(
            bin_edges=tuple(float(value) for value in self._free_gap_edges),
            counts=tuple(int(value) for value in self._gap_counts),
            normalization_denominator=total_gap_count,
            out_of_grid_count=self._gap_out_of_grid,
            expected_total_mass=1.0,
        )
        result = VariationalStreamResult(
            seed=self.seed,
            block_size=self.block_size,
            maximum_records=self.maximum_records,
            production_transition_count=self._total_batch_count,
            configuration_count=self._total_configuration_count,
            particle_count=self._particle_count,
            records=tuple(self._records),
            density=density,
            free_gap_distribution=free_gap_distribution,
        )
        self._finished_result = result
        return result

    def _seal_block(self) -> None:
        if len(self._records) >= self.maximum_records:
            raise RuntimeError("maximum_records is too small for the observed stream")
        if (
            self._batch_count < 1
            or self._configuration_count < 1
            or self._particle_count is None
            or self._first_step is None
            or self._last_step is None
        ):
            raise RuntimeError("cannot seal an empty variational block")
        values = self._scalar_sums / self._batch_count
        truncated_gradient = tuple(
            _TruncatedGradientEstimate(
                epsilon=float(epsilon),
                unconditional_t_grad=float(self._truncated_t_grad_sums[index] / self._batch_count),
                excluded_probability=float(
                    self._excluded_probability_sums[index] / self._batch_count
                ),
            )
            for index, epsilon in enumerate(self._cutoff_epsilons)
        )
        means = VariationalEnsembleMeans(
            t_local=float(values[0]),
            trap=float(values[1]),
            e_local=float(values[2]),
            r2=float(values[3]),
            weighted_free_gap=float(values[4]),
            truncated_gradient=truncated_gradient,
        )
        self._records.append(
            _VariationalBlockRecord(
                seed=self.seed,
                first_step=self._first_step,
                last_step=self._last_step,
                batch_count=self._batch_count,
                configuration_count=self._configuration_count,
                particle_count=self._particle_count,
                means=means,
            )
        )
        self._first_step = None
        self._batch_count = 0
        self._configuration_count = 0
        self._scalar_sums.fill(0.0)
        self._truncated_t_grad_sums.fill(0.0)
        self._excluded_probability_sums.fill(0.0)


class VariationalObserver:
    """Streaming VMC observer that reuses each event's evaluated guide state."""

    def __init__(
        self,
        *,
        center: float,
        rod_length: float,
        accumulator: VariationalStreamingAccumulator,
    ) -> None:
        self._center = float(center)
        self._rod_length = float(rod_length)
        self._accumulator = accumulator

    def record_vmc_transition(self, event: VMCTransitionEvent) -> None:
        """Consume one production event without a duplicate guide evaluation."""
        positions, observables = _observables_from_event(
            event,
            center=self._center,
            rod_length=self._rod_length,
            cutoff_epsilons=self._accumulator.cutoff_epsilons,
        )
        self._accumulator.observe(
            step=event.production_step,
            positions=positions,
            observables=observables,
        )

    def finish(self) -> VariationalStreamResult:
        return self._accumulator.finish()


def _observables_from_event(
    event: VMCTransitionEvent,
    *,
    center: float,
    rod_length: float,
    cutoff_epsilons: FloatArray,
) -> tuple[FloatArray, _VariationalBatchObservables]:
    observables = _compose_variational_observables(
        event.positions,
        event.gradients,
        event.local_energies,
        valid_mask=event.valid,
        center=center,
        rod_length=rod_length,
        cutoff_epsilons=cutoff_epsilons,
    )
    return np.asarray(event.positions, dtype=float), observables


def _validated_edges(values: FloatArray, name: str) -> FloatArray:
    edges = np.array(values, dtype=float, copy=True)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
    ):
        raise ValueError(f"{name} must be finite, one-dimensional, and strictly increasing")
    edges.setflags(write=False)
    return edges


def _validated_cutoffs(values: FloatArray) -> FloatArray:
    cutoffs = np.array(values, dtype=float, copy=True)
    if (
        cutoffs.ndim != 1
        or cutoffs.size < 1
        or not np.all(np.isfinite(cutoffs))
        or np.any(cutoffs <= 0.0)
        or not np.all(np.diff(cutoffs) > 0.0)
    ):
        raise ValueError("cutoff_epsilons must be finite, positive, and strictly increasing")
    cutoffs.setflags(write=False)
    return cutoffs


def _histogram_counts(values: FloatArray, edges: FloatArray) -> tuple[NDArray[np.int64], int]:
    samples = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(samples)):
        raise ValueError("histogram samples must be finite")
    in_grid = (samples >= edges[0]) & (samples <= edges[-1])
    counts, _ = np.histogram(samples[in_grid], bins=edges)
    return np.asarray(counts, dtype=np.int64), int(np.count_nonzero(~in_grid))
