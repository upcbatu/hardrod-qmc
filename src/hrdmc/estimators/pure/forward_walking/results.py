from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LagValue = float | FloatArray

PureWalkingStatus = Literal[
    "accepted",
    "schema_invalid",
    "plateau_unresolved",
    "insufficient_samples",
    "no_blocks",
]

PURE_STATUS_ACCEPTED: PureWalkingStatus = "accepted"
PURE_STATUS_SCHEMA_INVALID: PureWalkingStatus = "schema_invalid"
PURE_STATUS_PLATEAU_UNRESOLVED: PureWalkingStatus = "plateau_unresolved"
PURE_STATUS_INSUFFICIENT_SAMPLES: PureWalkingStatus = "insufficient_samples"
PURE_STATUS_NO_BLOCKS: PureWalkingStatus = "no_blocks"

SCHEMA_VALID = "schema_valid"
SCHEMA_INVALID = "schema_invalid"

PLATEAU_RESOLVED = "plateau_resolved"
PLATEAU_UNRESOLVED = "plateau_unresolved"
PLATEAU_NO_BLOCKS = "no_blocks"
PLATEAU_INSUFFICIENT_BLOCKS = "insufficient_blocks"
PLATEAU_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM = "effective_sample_count_below_minimum"

GENEALOGY_NOT_EVALUATED = "not_evaluated"
GENEALOGY_EFFECTIVE_SAMPLE_COUNT_BELOW_MINIMUM = "genealogy_effective_sample_count_below_minimum"
GENEALOGY_SOURCE_FAMILY_DOMINANCE = "source_family_dominance"
GENEALOGY_SUPPORT_ACCEPTED = "genealogy_support_accepted"


@dataclass(frozen=True)
class TransportedLagResult:
    observable: str
    lag_steps: tuple[int, ...]
    lag_unit: str
    values_by_lag: dict[int, LagValue]
    stderr_by_lag: dict[int, LagValue]
    block_count_by_lag: dict[int, int]
    block_weight_ess_min_by_lag: dict[int, float]
    block_weight_ess_mean_by_lag: dict[int, float]
    block_source_ancestor_ess_min_by_lag: dict[int, float]
    block_source_ancestor_ess_mean_by_lag: dict[int, float]
    block_unique_source_ancestor_min_by_lag: dict[int, int]
    block_max_source_family_fraction_by_lag: dict[int, float]
    variance_inflation_by_lag: dict[int, float]
    rms_radius_by_lag: dict[int, float] | None
    rms_radius_stderr_by_lag: dict[int, float] | None
    plateau_value: LagValue | None
    plateau_stderr: LagValue | None
    rms_radius: float | None
    rms_radius_stderr: float | None
    bias_bracket: tuple[LagValue, LagValue] | None
    plateau_status: str
    schema_status: str
    genealogy_status: str
    genealogy_diagnostics: dict[str, object]
    plateau_diagnostics: dict[str, object]
    metadata: dict[str, object]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "observable": self.observable,
            "lag_steps": list(self.lag_steps),
            "lag_unit": self.lag_unit,
            "values_by_lag": _json_lag_values(self.values_by_lag),
            "stderr_by_lag": _json_lag_values(self.stderr_by_lag),
            "block_count_by_lag": self.block_count_by_lag,
            "block_weight_ess_min_by_lag": self.block_weight_ess_min_by_lag,
            "block_weight_ess_mean_by_lag": self.block_weight_ess_mean_by_lag,
            "block_source_ancestor_ess_min_by_lag": (self.block_source_ancestor_ess_min_by_lag),
            "block_source_ancestor_ess_mean_by_lag": (self.block_source_ancestor_ess_mean_by_lag),
            "block_unique_source_ancestor_min_by_lag": (
                self.block_unique_source_ancestor_min_by_lag
            ),
            "block_max_source_family_fraction_by_lag": (
                self.block_max_source_family_fraction_by_lag
            ),
            "variance_inflation_by_lag": self.variance_inflation_by_lag,
            "rms_radius_by_lag": self.rms_radius_by_lag,
            "rms_radius_stderr_by_lag": self.rms_radius_stderr_by_lag,
            "plateau_value": _json_value(self.plateau_value),
            "plateau_stderr": _json_value(self.plateau_stderr),
            "rms_radius": self.rms_radius,
            "rms_radius_stderr": self.rms_radius_stderr,
            "bias_bracket": None
            if self.bias_bracket is None
            else (_json_value(self.bias_bracket[0]), _json_value(self.bias_bracket[1])),
            "plateau_status": self.plateau_status,
            "schema_status": self.schema_status,
            "genealogy_status": self.genealogy_status,
            "genealogy_diagnostics": _json_dict(self.genealogy_diagnostics),
            "plateau_diagnostics": _json_dict(self.plateau_diagnostics),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PureWalkingResult:
    status: PureWalkingStatus
    observable_results: dict[str, TransportedLagResult]
    metadata: dict[str, object]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observable_results": {
                name: result.to_summary_dict() for name, result in self.observable_results.items()
            },
            "metadata": self.metadata,
        }


def _json_lag_values(values: dict[int, LagValue]) -> dict[int, object]:
    return {lag: _json_value(value) for lag, value in values.items()}


def _json_value(value: LagValue | None) -> object:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_dict(values: dict[str, object]) -> dict[str, object]:
    return {key: _json_nested(value) for key, value in values.items()}


def _json_nested(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_nested(item) for item in value]
    return value
