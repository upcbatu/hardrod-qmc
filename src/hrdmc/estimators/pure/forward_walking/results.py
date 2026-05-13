from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LagValue = float | FloatArray

PureWalkingStatus = Literal[
    "PURE_WALKING_GO",
    "PURE_WALKING_SCHEMA_NO_GO",
    "PURE_WALKING_PLATEAU_NO_GO",
    "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO",
    "PURE_WALKING_NO_BLOCKS_NO_GO",
]


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
    variance_inflation_by_lag: dict[int, float]
    paper_rms_radius_by_lag: dict[int, float] | None
    paper_rms_radius_stderr_by_lag: dict[int, float] | None
    plateau_value: LagValue | None
    plateau_stderr: LagValue | None
    paper_rms_radius: float | None
    paper_rms_radius_stderr: float | None
    bias_bracket: tuple[LagValue, LagValue] | None
    plateau_status: str
    schema_status: str
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
            "variance_inflation_by_lag": self.variance_inflation_by_lag,
            "paper_rms_radius_by_lag": self.paper_rms_radius_by_lag,
            "paper_rms_radius_stderr_by_lag": self.paper_rms_radius_stderr_by_lag,
            "plateau_value": _json_value(self.plateau_value),
            "plateau_stderr": _json_value(self.plateau_stderr),
            "paper_rms_radius": self.paper_rms_radius,
            "paper_rms_radius_stderr": self.paper_rms_radius_stderr,
            "bias_bracket": None
            if self.bias_bracket is None
            else (_json_value(self.bias_bracket[0]), _json_value(self.bias_bracket[1])),
            "plateau_status": self.plateau_status,
            "schema_status": self.schema_status,
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
                name: result.to_summary_dict()
                for name, result in self.observable_results.items()
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
