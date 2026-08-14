from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.statistics.density import (
    fixed_shell_cell_average,
    shell_peak_profile,
    unit_mass_shell_average,
)
from hrdmc.statistics.equivalence import (
    SimultaneousPairwiseEquivalenceResult,
    SimultaneousPairwiseNormEquivalenceResult,
    simultaneous_pairwise_equivalence,
    simultaneous_pairwise_norm_equivalence,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _ShellComparison:
    boundaries: FloatArray
    centers: FloatArray
    anchor_envelope: FloatArray
    candidate_envelope: FloatArray
    anchor_peak_positions: FloatArray
    candidate_peak_positions: FloatArray
    peak_position_shift_in_cell_widths: FloatArray
    anchor_peak_amplitudes: FloatArray
    candidate_peak_amplitudes: FloatArray
    peak_amplitude_relative_difference: FloatArray

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_definition": "fixed unit-mass cells from the anchor density",
            "cell_scope": "all N cells on the exact common histogram grid",
            "cell_count": int(self.centers.size),
            "tail_cell_indices": [0, int(self.centers.size - 1)],
            "tail_cell_caveat": (
                "the first and last cells terminate at finite histogram edges and "
                "their averaged values are grid-truncation diagnostics"
            ),
            "boundaries": self.boundaries.tolist(),
            "centers": self.centers.tolist(),
            "anchor_envelope": self.anchor_envelope.tolist(),
            "candidate_envelope": self.candidate_envelope.tolist(),
            "anchor_peak_positions": self.anchor_peak_positions.tolist(),
            "candidate_peak_positions": self.candidate_peak_positions.tolist(),
            "peak_position_shift_in_cell_widths": (
                self.peak_position_shift_in_cell_widths.tolist()
            ),
            "anchor_peak_amplitudes": self.anchor_peak_amplitudes.tolist(),
            "candidate_peak_amplitudes": self.candidate_peak_amplitudes.tolist(),
            "peak_amplitude_relative_difference": (
                self.peak_amplitude_relative_difference.tolist()
            ),
            "maximum_absolute_peak_position_shift_in_cell_widths": float(
                np.max(np.abs(self.peak_position_shift_in_cell_widths))
            ),
            "maximum_absolute_peak_amplitude_relative_difference": float(
                np.max(np.abs(self.peak_amplitude_relative_difference))
            ),
            "decision_role": (
                "descriptive only; exact-grid paired density L2 carries the vector "
                "observable equivalence decision"
            ),
        }


@dataclass(frozen=True)
class ForwardWalkingSensitivityResult:
    r2_equivalence: SimultaneousPairwiseEquivalenceResult
    rms_equivalence: SimultaneousPairwiseEquivalenceResult
    density_equivalence: SimultaneousPairwiseNormEquivalenceResult
    envelope_relative_l2_by_seed: FloatArray
    r2_relative_margin: float
    rms_relative_margin: float
    density_relative_l2_margin: float
    anchor_r2_scale: float
    anchor_rms_scale: float
    aggregate_density_relative_l2: float
    aggregate_envelope_relative_l2: float
    shell_comparison: _ShellComparison

    @property
    def equivalent(self) -> bool:
        return bool(
            self.r2_equivalence.equivalent
            and self.rms_equivalence.equivalent
            and self.density_equivalence.equivalent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "r2": {
                **self.r2_equivalence.to_dict(),
                "relative_equivalence_margin": self.r2_relative_margin,
                "anchor_scale": self.anchor_r2_scale,
                "simultaneous_relative_upper_bound": (
                    self.r2_equivalence.simultaneous_upper_bound / self.anchor_r2_scale
                ),
            },
            "rms_radius": {
                **self.rms_equivalence.to_dict(),
                "relative_equivalence_margin": self.rms_relative_margin,
                "anchor_scale": self.anchor_rms_scale,
                "simultaneous_relative_upper_bound": (
                    self.rms_equivalence.simultaneous_upper_bound / self.anchor_rms_scale
                ),
            },
            "density": {
                **self.density_equivalence.to_dict(),
                "aggregate_relative_l2": self.aggregate_density_relative_l2,
                "grid_weighting": "exact histogram bin widths",
                "reported_norm": "bin-width-weighted relative L2",
                "weighted_l1_policy": (
                    "not reported because the declared sensitivity criterion is L2"
                ),
            },
            "fixed_cell_envelope": {
                "aggregate_relative_l2": self.aggregate_envelope_relative_l2,
                "paired_seed_relative_l2": self.envelope_relative_l2_by_seed.tolist(),
                "mean_paired_seed_relative_l2": float(np.mean(self.envelope_relative_l2_by_seed)),
                "maximum_paired_seed_relative_l2": float(np.max(self.envelope_relative_l2_by_seed)),
                "cell_weighting": "fixed anchor shell-cell widths",
                "decision_role": "descriptive_only",
                "reason": (
                    "the common shell boundaries are estimated from the same anchor "
                    "replicates, so an independent-replicate t bound is not claimed"
                ),
            },
            "shell_peaks": self.shell_comparison.to_dict(),
            "extrapolation_policy": (
                "no binwise density extrapolation; this is a paired sensitivity bound"
            ),
        }


def classify_fw_sensitivity_status(
    *,
    input_quality_accepted: bool,
    density_grid_compatible: bool,
    plateau_resolved: bool,
    genealogy_supported: bool,
    observables_equivalent: bool,
    has_warnings: bool,
) -> str:
    """Classify FW sensitivity with scientific failure precedence."""
    if not input_quality_accepted:
        return "input_quality_unresolved"
    if not density_grid_compatible:
        return "density_grid_incompatible"
    if not plateau_resolved:
        return "plateau_unresolved"
    if not genealogy_supported:
        return "genealogy_unresolved"
    if not observables_equivalent:
        return "observable_sensitive"
    if has_warnings:
        return "accepted_with_warnings"
    return "accepted"


def analyze_fw_observable_sensitivity(
    *,
    anchor_r2_by_seed: FloatArray,
    candidate_r2_by_seed: FloatArray,
    bin_edges: FloatArray,
    anchor_density_by_seed: FloatArray,
    candidate_density_by_seed: FloatArray,
    particle_count: int,
    rms_relative_margin: float = 0.001,
    density_relative_l2_margin: float = 0.03,
    confidence_level: float = 0.95,
    density_normalization_atol: float = 0.005,
) -> ForwardWalkingSensitivityResult:
    """Compare paired pure-FW scalar and density estimates across treatments."""
    anchor_r2, candidate_r2, edges, anchor_density, candidate_density = (
        _validated_sensitivity_inputs(
            anchor_r2_by_seed=anchor_r2_by_seed,
            candidate_r2_by_seed=candidate_r2_by_seed,
            bin_edges=bin_edges,
            anchor_density_by_seed=anchor_density_by_seed,
            candidate_density_by_seed=candidate_density_by_seed,
            particle_count=particle_count,
            density_normalization_atol=density_normalization_atol,
        )
    )
    _validate_sensitivity_margins(
        rms_relative_margin=rms_relative_margin,
        density_relative_l2_margin=density_relative_l2_margin,
        density_normalization_atol=density_normalization_atol,
    )
    widths = np.diff(edges)
    anchor_r2_scale = float(np.mean(anchor_r2))
    anchor_rms_by_seed = np.sqrt(anchor_r2)
    candidate_rms_by_seed = np.sqrt(candidate_r2)
    anchor_rms_scale = float(np.sqrt(anchor_r2_scale))
    # A symmetric R2 margin must use the smaller (downward) side of the
    # transformation R2=RMS^2 to remain conservative in both directions.
    r2_relative_margin = float(2.0 * rms_relative_margin - rms_relative_margin**2)
    r2_equivalence = simultaneous_pairwise_equivalence(
        np.column_stack((anchor_r2, candidate_r2)),
        equivalence_margin=r2_relative_margin * anchor_r2_scale,
        confidence_level=confidence_level,
    )
    rms_equivalence = simultaneous_pairwise_equivalence(
        np.column_stack((anchor_rms_by_seed, candidate_rms_by_seed)),
        equivalence_margin=rms_relative_margin * anchor_rms_scale,
        confidence_level=confidence_level,
    )
    anchor_density_mean = np.mean(anchor_density, axis=0)
    candidate_density_mean = np.mean(candidate_density, axis=0)
    density_equivalence = simultaneous_pairwise_norm_equivalence(
        np.stack((anchor_density, candidate_density), axis=1),
        feature_weights=widths,
        scale_vector=anchor_density_mean,
        equivalence_margin=density_relative_l2_margin,
        confidence_level=confidence_level,
    )
    aggregate_density_relative_l2 = _weighted_relative_l2(
        candidate_density_mean,
        anchor_density_mean,
        widths,
    )
    (
        envelope_relative_l2_by_seed,
        aggregate_envelope_relative_l2,
        shell_comparison,
    ) = _shell_sensitivity(
        edges=edges,
        anchor_density=anchor_density,
        candidate_density=candidate_density,
        particle_count=particle_count,
        normalization_atol=density_normalization_atol,
    )
    return ForwardWalkingSensitivityResult(
        r2_equivalence=r2_equivalence,
        rms_equivalence=rms_equivalence,
        density_equivalence=density_equivalence,
        envelope_relative_l2_by_seed=envelope_relative_l2_by_seed,
        r2_relative_margin=r2_relative_margin,
        rms_relative_margin=float(rms_relative_margin),
        density_relative_l2_margin=float(density_relative_l2_margin),
        anchor_r2_scale=anchor_r2_scale,
        anchor_rms_scale=anchor_rms_scale,
        aggregate_density_relative_l2=aggregate_density_relative_l2,
        aggregate_envelope_relative_l2=aggregate_envelope_relative_l2,
        shell_comparison=shell_comparison,
    )


def _validated_sensitivity_inputs(
    *,
    anchor_r2_by_seed: FloatArray,
    candidate_r2_by_seed: FloatArray,
    bin_edges: FloatArray,
    anchor_density_by_seed: FloatArray,
    candidate_density_by_seed: FloatArray,
    particle_count: int,
    density_normalization_atol: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    anchor_r2 = _validated_r2(anchor_r2_by_seed, name="anchor_r2_by_seed")
    candidate_r2 = _validated_r2(candidate_r2_by_seed, name="candidate_r2_by_seed")
    if anchor_r2.shape != candidate_r2.shape:
        raise ValueError("anchor and candidate R2 arrays must have identical shapes")
    if anchor_r2.size < 2:
        raise ValueError("FW sensitivity requires at least two paired seeds")
    edges = np.asarray(bin_edges, dtype=np.float64)
    anchor_density = _validated_density_rows(
        anchor_density_by_seed,
        edges=edges,
        name="anchor_density_by_seed",
    )
    candidate_density = _validated_density_rows(
        candidate_density_by_seed,
        edges=edges,
        name="candidate_density_by_seed",
    )
    if anchor_density.shape != candidate_density.shape:
        raise ValueError("anchor and candidate density arrays must have identical shapes")
    if anchor_density.shape[0] != anchor_r2.size:
        raise ValueError("R2 and density arrays must contain the same paired seeds")
    if (
        not isinstance(particle_count, int)
        or isinstance(particle_count, bool)
        or particle_count < 3
    ):
        raise ValueError("particle_count must be an integer of at least three")
    widths = np.diff(edges)
    for name, rows in (("anchor", anchor_density), ("candidate", candidate_density)):
        if not np.allclose(
            rows @ widths,
            float(particle_count),
            rtol=0.0,
            atol=density_normalization_atol,
        ):
            raise ValueError(f"{name} seed densities do not integrate to particle_count")
    return anchor_r2, candidate_r2, edges, anchor_density, candidate_density


def _validate_sensitivity_margins(
    *,
    rms_relative_margin: float,
    density_relative_l2_margin: float,
    density_normalization_atol: float,
) -> None:
    for name, value in (
        ("rms_relative_margin", rms_relative_margin),
        ("density_relative_l2_margin", density_relative_l2_margin),
        ("density_normalization_atol", density_normalization_atol),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if rms_relative_margin >= 1.0:
        raise ValueError("rms_relative_margin must be smaller than one")


def _shell_sensitivity(
    *,
    edges: FloatArray,
    anchor_density: FloatArray,
    candidate_density: FloatArray,
    particle_count: int,
    normalization_atol: float,
) -> tuple[FloatArray, float, _ShellComparison]:
    anchor_density_mean = np.mean(anchor_density, axis=0)
    candidate_density_mean = np.mean(candidate_density, axis=0)
    anchor_unit_cells = unit_mass_shell_average(
        edges,
        anchor_density_mean,
        particle_count=particle_count,
        normalization_atol=normalization_atol,
    )
    all_shell_boundaries = np.concatenate(
        (
            edges[:1],
            anchor_unit_cells.boundaries,
            edges[-1:],
        )
    )
    anchor_envelope = fixed_shell_cell_average(
        edges,
        anchor_density_mean,
        boundaries=all_shell_boundaries,
        replicate_densities=anchor_density,
    )
    candidate_envelope = fixed_shell_cell_average(
        edges,
        candidate_density_mean,
        boundaries=all_shell_boundaries,
        replicate_densities=candidate_density,
    )
    assert anchor_envelope.replicate_values is not None
    assert candidate_envelope.replicate_values is not None
    cell_widths = np.diff(all_shell_boundaries)
    envelope_scale_norm = float(np.sqrt(np.sum(anchor_envelope.values**2 * cell_widths)))
    if not np.isfinite(envelope_scale_norm) or envelope_scale_norm <= 0.0:
        raise ValueError("anchor fixed-cell envelope must have a positive weighted norm")
    envelope_relative_l2_by_seed = (
        np.sqrt(
            np.sum(
                (candidate_envelope.replicate_values - anchor_envelope.replicate_values) ** 2
                * cell_widths,
                axis=1,
            )
        )
        / envelope_scale_norm
    )
    aggregate_envelope_relative_l2 = _weighted_relative_l2(
        candidate_envelope.values,
        anchor_envelope.values,
        cell_widths,
    )
    anchor_peaks = shell_peak_profile(
        edges,
        anchor_density_mean,
        boundaries=all_shell_boundaries,
    )
    candidate_peaks = shell_peak_profile(
        edges,
        candidate_density_mean,
        boundaries=all_shell_boundaries,
    )
    amplitude_scale = np.maximum(anchor_peaks.amplitudes, np.finfo(np.float64).tiny)
    shell_comparison = _ShellComparison(
        boundaries=all_shell_boundaries.copy(),
        centers=anchor_envelope.centers.copy(),
        anchor_envelope=anchor_envelope.values.copy(),
        candidate_envelope=candidate_envelope.values.copy(),
        anchor_peak_positions=anchor_peaks.positions.copy(),
        candidate_peak_positions=candidate_peaks.positions.copy(),
        peak_position_shift_in_cell_widths=(
            (candidate_peaks.positions - anchor_peaks.positions) / cell_widths
        ),
        anchor_peak_amplitudes=anchor_peaks.amplitudes.copy(),
        candidate_peak_amplitudes=candidate_peaks.amplitudes.copy(),
        peak_amplitude_relative_difference=(
            (candidate_peaks.amplitudes - anchor_peaks.amplitudes) / amplitude_scale
        ),
    )
    return (
        np.asarray(envelope_relative_l2_by_seed, dtype=np.float64),
        aggregate_envelope_relative_l2,
        shell_comparison,
    )


def _validated_r2(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a one-dimensional array of positive finite values")
    return array


def _validated_density_rows(
    values: FloatArray,
    *,
    edges: FloatArray,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("bin_edges must be a finite one-dimensional array")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin_edges must be strictly increasing")
    if array.ndim != 2 or array.shape[1] != edges.size - 1:
        raise ValueError(f"{name} must have shape (seeds, bins)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _weighted_relative_l2(candidate: FloatArray, anchor: FloatArray, weights: FloatArray) -> float:
    numerator = float(np.sum((candidate - anchor) ** 2 * weights))
    denominator = float(np.sum(anchor**2 * weights))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("anchor vector must have positive weighted norm")
    return float(np.sqrt(numerator / denominator))
