from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from hrdmc.statistics.rank_diagnostics import rank_normalized_diagnostics
from hrdmc.statistics.timeseries import (
    trace_stationarity_diagnostics,
    trace_stationarity_result_to_dict,
)
from hrdmc.statistics.vector_equivalence import (
    unpaired_scalar_equivalence,
    unpaired_vector_equivalence,
)
from hrdmc.system.settings import TrappedCase
from hrdmc.validation.sampler_equivalence.kinetic import extrapolate_gradient_cutoff
from hrdmc.validation.sampler_equivalence.models import (
    VMC_CUTOFF_EPSILONS,
    VMC_VALIDATION_CASE_IDS,
    VMCValidationPolicy,
)
from hrdmc.validation.sampler_equivalence.seed import VMCSeedRun

SCALAR_OBSERVABLES = (
    "e_local",
    "t_local",
    "trap",
    "r2",
    "weighted_free_gap",
)
_QUALIFIED_STATUSES = frozenset({"accepted", "accepted_with_warnings"})
@dataclass(frozen=True)
class VMCAssessment:
    status: str
    payload: dict[str, Any]
def assess_vmc_case(
    case: TrappedCase,
    runs: list[VMCSeedRun],
    *,
    policy: VMCValidationPolicy,
) -> VMCAssessment:
    policy.validate()
    groups = _validated_groups(case, runs)
    sampler_payloads = {
        sampler: _assess_sampler(sampler, sampler_runs, policy=policy)
        for sampler, sampler_runs in groups.items()
    }
    equivalence = _assess_equivalence(groups, policy=policy)
    failures = []
    for sampler, payload in sampler_payloads.items():
        if not _is_qualified_status(payload["status"]):
            failures.append(f"{sampler}:{payload['status']}")
    if equivalence["status"] != "equivalent":
        failures.append(f"sampler_equivalence:{equivalence['status']}")
    has_warnings = any(
        payload["status"] == "accepted_with_warnings" for payload in sampler_payloads.values()
    )
    status = (
        ("accepted_with_warnings" if has_warnings else "accepted") if not failures else "unresolved"
    )
    objective_payload = {
        "kinetic_estimator_consistency": sampler_payloads["random_walk_metropolis"][
            "kinetic_estimator_consistency"
        ],
        "branching_free_sampler_equivalence": {
            "status": equivalence["status"],
            "claim_scope": (
                "TFM Offer objective 4: the production DMC local transition without "
                "branching reproduces the independent VMC distribution and compared "
                "observables"
            ),
        },
    }
    return VMCAssessment(
        status=status,
        payload={
            "status": status,
            "case_id": case.case_id,
            "policy": policy.to_dict(),
            "samplers": sampler_payloads,
            "sampler_equivalence": equivalence,
            "objectives": objective_payload,
            "failed_or_unresolved_checks": failures,
        },
    )
def _assess_sampler(
    sampler: str,
    runs: list[VMCSeedRun],
    *,
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    traces = {name: _trace_matrix(runs, name) for name in SCALAR_OBSERVABLES}
    cutoffs, cutoff_values, cutoff_excluded = _cutoff_matrices(runs)
    for index, epsilon in enumerate(cutoffs):
        traces[f"t_grad_cutoff_{epsilon:g}"] = cutoff_values[:, :, index]
    diagnostics = {
        name: _diagnose_trace_matrix(values, policy=policy) for name, values in traces.items()
    }
    cutoff = extrapolate_gradient_cutoff(cutoffs, cutoff_values, cutoff_excluded)
    t_local_by_seed = _seed_means(_trace_matrix(runs, "t_local"))
    paired_difference = np.asarray(t_local_by_seed) - np.asarray(cutoff.primary.intercept_by_seed)
    kinetic = _paired_kinetic_consistency(
        paired_difference,
        t_local_by_seed=t_local_by_seed,
        cutoff=cutoff,
        policy=policy,
    )
    histogram = _histogram_accounting(runs, policy=policy)
    kinetic_required = sampler == "random_walk_metropolis"
    diagnostics_pass = all(row["accepted"] for row in diagnostics.values())
    checks = {
        "diagnostics": diagnostics_pass,
        "histogram_accounting": histogram["accepted"],
    }
    if kinetic_required:
        checks["kinetic_consistency"] = kinetic["accepted"]
        checks["cutoff_monotonicity"] = cutoff.monotone_nonincreasing_with_epsilon
    has_warnings = any(row["status"] == "accepted_with_warnings" for row in diagnostics.values())
    status = (
        ("accepted_with_warnings" if has_warnings else "accepted")
        if all(checks.values())
        else "unresolved"
    )
    return {
        "status": status,
        "checks": checks,
        "chain_diagnostics": diagnostics,
        "gradient_cutoff_extrapolation": cutoff.to_dict(),
        "kinetic_estimator_consistency": kinetic,
        "kinetic_estimator_consistency_required": kinetic_required,
        "histogram_accounting": histogram,
        "seed_means": {name: _seed_means(values).tolist() for name, values in traces.items()},
        "density_by_seed": _profile_matrix(runs, "density").tolist(),
        "free_gap_distribution_by_seed": _profile_matrix(runs, "free_gap_distribution").tolist(),
    }
def _diagnose_trace_matrix(
    values: np.ndarray,
    *,
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    rank = rank_normalized_diagnostics(values)
    times = np.arange(values.shape[1], dtype=float)
    stationarity = [
        trace_stationarity_diagnostics(times, values[index]) for index in range(values.shape[0])
    ]
    finite_per_seed_ess = all(np.isfinite(value) for value in rank.bulk_ess_per_chain)
    per_seed_ok = finite_per_seed_ess and all(
        value >= policy.per_seed_ess_minimum for value in rank.bulk_ess_per_chain
    )
    finite_mcse = all(np.isfinite(result.blocking_stderr) for result in stationarity)
    required_failures: list[str] = []
    if not np.isfinite(rank.split_rhat):
        required_failures.append("nonfinite_split_rhat")
    elif rank.split_rhat >= policy.rhat_limit:
        required_failures.append("split_rhat_at_or_above_limit")
    if not np.isfinite(rank.bulk_ess):
        required_failures.append("nonfinite_total_bulk_ess")
    elif rank.bulk_ess < policy.total_bulk_ess_minimum:
        required_failures.append("total_bulk_ess_below_minimum")
    if not finite_per_seed_ess:
        required_failures.append("nonfinite_per_seed_bulk_ess")
    elif not per_seed_ok:
        required_failures.append("per_seed_bulk_ess_below_minimum")
    if not finite_mcse:
        required_failures.append("nonfinite_autocorrelation_aware_mcse")
    warning_records = _custom_stationarity_warning_records(stationarity)
    warning_reasons: dict[str, int] = {}
    for record in warning_records:
        for reason in record["reasons"]:
            warning_reasons[reason] = warning_reasons.get(reason, 0) + 1
    warning_count = len(warning_records)
    warning_reason_count = sum(warning_reasons.values())
    accepted = not required_failures
    status = (
        ("accepted_with_warnings" if warning_count else "accepted")
        if accepted
        else "insufficient_information"
    )
    return {
        "accepted": accepted,
        "status": status,
        "finite_autocorrelation_aware_mcse": finite_mcse,
        "required_diagnostic_failures": required_failures,
        "warning_count": warning_count,
        "warning_reason_count": warning_reason_count,
        "warning_reasons": warning_reasons,
        "warnings_by_seed": warning_records,
        "rank_normalized": rank.to_dict(),
        "stationarity": [trace_stationarity_result_to_dict(result) for result in stationarity],
    }
def _custom_stationarity_warning_records(stationarity: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed_index, result in enumerate(stationarity):
        reasons: list[str] = []
        if not result.trend_clean:
            reasons.append("autocorrelation_adjusted_trend_alert")
        if not result.cumulative_drift_clean:
            reasons.append("cumulative_drift_alert")
        if not result.blocking_clean:
            reasons.append("first_last_block_alert")
        if result.spread_warning:
            reasons.append("block_spread_alert")
        if not result.stationarity_clean and not reasons:
            reasons.append("custom_stationarity_alert")
        if reasons:
            records.append({"seed_index": seed_index, "reasons": reasons})
    return records
def _is_qualified_status(status: Any) -> bool:
    return str(status) in _QUALIFIED_STATUSES
def _paired_kinetic_consistency(
    differences: np.ndarray,
    *,
    t_local_by_seed: np.ndarray,
    cutoff: Any,
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    cutoff_usable, unusable_reasons = _cutoff_fit_is_usable(cutoff)
    if differences.size < 2:
        unusable_reasons.append("fewer_than_two_independent_seed_differences")
    if not np.all(np.isfinite(differences)):
        unusable_reasons.append("nonfinite_seed_differences")
    if not np.all(np.isfinite(t_local_by_seed)):
        unusable_reasons.append("nonfinite_t_local_seed_means")
    if unusable_reasons or not cutoff_usable:
        return {
            "accepted": False,
            "status": "insufficient_information",
            "unusable_reasons": unusable_reasons,
        }
    critical = float(
        student_t.ppf(
            1.0 - (1.0 - policy.confidence_level) / (2.0 * 2.0),
            differences.size - 1,
        )
    )
    mean_difference = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
    stochastic_half_width = critical * standard_error
    criterion = abs(mean_difference) + stochastic_half_width + cutoff.model_spread
    scale = 0.5 * (abs(float(np.mean(t_local_by_seed))) + abs(float(cutoff.intercept)))
    margin = policy.kinetic_relative_margin * scale
    if not np.isfinite(criterion) or not np.isfinite(scale) or scale <= 0.0:
        return {
            "accepted": False,
            "status": "insufficient_information",
            "unusable_reasons": ["nonfinite_bound_or_nonpositive_scale"],
        }
    accepted = bool(criterion < margin)
    if accepted:
        status = "accepted"
    elif abs(mean_difference) > margin:
        status = "not_equivalent"
    else:
        status = "insufficient_information"
    return {
        "accepted": accepted,
        "status": status,
        "comparison_design": "paired_by_seed_same_configuration_stream",
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "critical_value": critical,
        "familywise_stochastic_half_width": stochastic_half_width,
        "cutoff_model_spread": cutoff.model_spread,
        "sensitivity_augmented_absolute_criterion": criterion,
        "interpretation_scope": (
            "the Student-t component has familywise stochastic coverage; "
            "cutoff-model spread is a predeclared sensitivity penalty, not a "
            "coverage-calibrated truncation-bias bound"
        ),
        "kinetic_scale": scale,
        "absolute_margin": margin,
        "relative_margin": policy.kinetic_relative_margin,
    }
def _assess_equivalence(
    groups: dict[str, list[VMCSeedRun]],
    *,
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    first = groups["random_walk_metropolis"]
    second = groups["branching_free_mala"]
    margins = {
        "e_local": policy.energy_relative_margin,
        "t_local": policy.kinetic_relative_margin,
        "trap": policy.trap_relative_margin,
        "r2": policy.r2_relative_margin,
        "weighted_free_gap": policy.weighted_free_gap_relative_margin,
    }
    scalar: dict[str, Any] = {}
    for name, relative_margin in margins.items():
        first_values = _seed_means(_trace_matrix(first, name))
        second_values = _seed_means(_trace_matrix(second, name))
        scale = 0.5 * (abs(float(np.mean(first_values))) + abs(float(np.mean(second_values))))
        result = unpaired_scalar_equivalence(
            first_values,
            second_values,
            practical_margin=relative_margin * scale,
            familywise_confidence=policy.confidence_level,
            family_size=len(margins) + 2,
        )
        scalar[name] = {
            **result.to_dict(),
            "relative_margin": relative_margin,
            "normalization_scale": scale,
        }
    vectors: dict[str, Any] = {}
    for name, relative_margin, rng_seed in (
        ("density", policy.density_relative_l2_margin, 902_101),
        (
            "free_gap_distribution",
            policy.gap_distribution_relative_l2_margin,
            902_102,
        ),
    ):
        first_profiles = _profile_matrix(first, name)
        second_profiles = _profile_matrix(second, name)
        record = getattr(first[0].estimates, name)
        widths = np.diff(np.asarray(record.bin_edges, dtype=float))
        scale_profile = 0.5 * (np.mean(first_profiles, axis=0) + np.mean(second_profiles, axis=0))
        result = unpaired_vector_equivalence(
            first_profiles,
            second_profiles,
            feature_weights=widths,
            scale_profile=scale_profile,
            practical_margin=relative_margin,
            rng_seed=rng_seed,
            bootstrap_replicates=10_000,
            familywise_confidence=policy.confidence_level,
            family_size=len(margins) + 2,
        )
        vectors[name] = result.to_dict()
    statuses = [row["status"] for row in scalar.values()] + [
        row["status"] for row in vectors.values()
    ]
    if any(value == "not_equivalent" for value in statuses):
        status = "not_equivalent"
    elif any(value == "insufficient_information" for value in statuses):
        status = "insufficient_information"
    else:
        status = "equivalent"
    return {
        "status": status,
        "design": "unpaired_independent_sampler_seed_groups",
        "interpretation_scope": (
            "insufficient_information means the observed sampler discrepancy is "
            "within its practical margin but uncertainty crosses that margin; it "
            "does not establish different target distributions. not_equivalent is "
            "reserved for an observed central discrepancy above the margin"
        ),
        "scalar": scalar,
        "profiles": vectors,
    }
def _validated_groups(
    case: TrappedCase,
    runs: list[VMCSeedRun],
) -> dict[str, list[VMCSeedRun]]:
    if case.case_id not in VMC_VALIDATION_CASE_IDS:
        raise ValueError(f"VMC validation is prospectively limited to {VMC_VALIDATION_CASE_IDS}")
    supported = {"random_walk_metropolis", "branching_free_mala"}
    observed = {run.sampler for run in runs}
    unknown = sorted(observed - supported)
    if unknown:
        raise ValueError(f"assessment contains unsupported sampler labels: {unknown}")
    groups = {
        sampler: sorted(
            [run for run in runs if run.sampler == sampler],
            key=lambda run: run.seed,
        )
        for sampler in ("random_walk_metropolis", "branching_free_mala")
    }
    for sampler, rows in groups.items():
        if len(rows) != 5:
            raise ValueError(f"assessment requires exactly five {sampler} seed runs")
        if any(row.case_id != case.case_id for row in rows):
            raise ValueError("assessment contains a run from another case")
        if len({row.seed for row in rows}) != len(rows):
            raise ValueError("sampler seed runs must have unique seeds")
    if not {row.seed for row in groups["random_walk_metropolis"]}.isdisjoint(
        row.seed for row in groups["branching_free_mala"]
    ):
        raise ValueError("RWM and MALA production seed streams must be disjoint")
    return groups
def _trace_matrix(runs: list[VMCSeedRun], name: str) -> np.ndarray:
    return _trace_matrix_from_streams([run.estimates for run in runs], name)
def _trace_matrix_from_streams(streams: list[Any], name: str) -> np.ndarray:
    rows = [[float(getattr(record.means, name)) for record in stream.records] for stream in streams]
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"observable {name} has a malformed trace matrix")
    return array
def _cutoff_matrices(
    runs: list[VMCSeedRun],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cutoff_matrices_from_streams([run.estimates for run in runs])
def _cutoff_matrices_from_streams(
    streams: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = np.asarray(VMC_CUTOFF_EPSILONS, dtype=float)
    for stream in streams:
        if not stream.records:
            raise ValueError("cutoff assessment requires at least one production block")
        for record in stream.records:
            observed = np.asarray(
                [row.epsilon for row in record.means.truncated_gradient],
                dtype=float,
            )
            if not np.array_equal(observed, expected):
                raise ValueError(
                    f"every VMC block must use the prospective cutoff ladder {VMC_CUTOFF_EPSILONS}"
                )
    epsilon = expected
    values = np.asarray(
        [
            [
                [row.unconditional_t_grad for row in record.means.truncated_gradient]
                for record in stream.records
            ]
            for stream in streams
        ],
        dtype=float,
    )
    excluded = np.asarray(
        [
            [
                [row.excluded_probability for row in record.means.truncated_gradient]
                for record in stream.records
            ]
            for stream in streams
        ],
        dtype=float,
    )
    return epsilon, values, excluded
def _seed_means(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(values, axis=1), dtype=float)
def _profile_matrix(runs: list[VMCSeedRun], name: str) -> np.ndarray:
    return _profile_matrix_from_streams([run.estimates for run in runs], name)
def _profile_matrix_from_streams(streams: list[Any], name: str) -> np.ndarray:
    profiles = np.asarray(
        [getattr(stream, name).density for stream in streams],
        dtype=float,
    )
    if profiles.ndim != 2 or not np.all(np.isfinite(profiles)):
        raise ValueError(f"profile {name} has malformed seed values")
    return profiles
def _histogram_accounting(
    runs: list[VMCSeedRun],
    *,
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    return _histogram_accounting_from_streams(
        [run.estimates for run in runs],
        seeds=[run.seed for run in runs],
        policy=policy,
    )
def _histogram_accounting_from_streams(
    streams: list[Any],
    *,
    seeds: list[int],
    policy: VMCValidationPolicy,
) -> dict[str, Any]:
    if len(streams) != len(seeds):
        raise ValueError("histogram streams and seed identifiers must have equal length")
    rows = []
    accepted = True
    for seed, stream in zip(seeds, streams, strict=True):
        row: dict[str, Any] = {"seed": seed}
        for name in ("density", "free_gap_distribution"):
            record = getattr(stream, name)
            expected = record.expected_total_mass
            integral_error = abs(record.in_grid_mass + record.out_of_grid_mass - expected)
            ok = bool(
                integral_error <= policy.histogram_normalization_tolerance
                and record.out_of_grid_mass <= policy.histogram_out_of_grid_mass_limit
            )
            row[name] = {
                "accepted": ok,
                "integral_accounting_error": integral_error,
                "out_of_grid_mass": record.out_of_grid_mass,
                "expected_total_mass": expected,
            }
            accepted = accepted and ok
        rows.append(row)
    return {"accepted": accepted, "seed_checks": rows}
def _cutoff_fit_is_usable(cutoff: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(cutoff.monotone_nonincreasing_with_epsilon):
        reasons.append("cutoff_sequence_not_monotone")
    finite_values = np.asarray(
        [
            cutoff.intercept,
            cutoff.model_spread,
            cutoff.intercept_standard_error,
            *cutoff.primary.intercept_by_seed,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(finite_values)):
        reasons.append("cutoff_fit_contains_nonfinite_values")
    if cutoff.model_spread < 0.0 or cutoff.intercept_standard_error < 0.0:
        reasons.append("cutoff_uncertainty_is_negative")
    return not reasons, reasons
