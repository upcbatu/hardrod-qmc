from __future__ import annotations

import numpy as np

from hrdmc.analysis import CHAIN_ACCEPTED, CHAIN_SPREAD_WARNING
from hrdmc.estimators.pure.forward_walking.results import PURE_STATUS_ACCEPTED

ACCEPTED_ENERGY_STATIONARITY = {CHAIN_ACCEPTED, CHAIN_SPREAD_WARNING}


def energy_validation_status(stationarity: dict) -> str:
    if not bool(stationarity.get("density_accounting_clean", False)):
        return "density_normalization_mismatch"
    if not bool(stationarity.get("valid_finite_clean", False)):
        return "nonfinite_samples"
    population_status = str(stationarity.get("population_weight_status", ""))
    if population_status != "accepted":
        return population_status or "weight_status_unavailable"
    energy_stationarity = str(stationarity.get("stationarity_energy", ""))
    if energy_stationarity not in ACCEPTED_ENERGY_STATIONARITY:
        return energy_stationarity or "stationarity_unavailable"
    return "accepted"


def pure_fw_validation_status(pure_summary: dict) -> str:
    return str(pure_summary.get("status", "not_evaluated"))


def benchmark_validation_status(*, energy_status: str, pure_status: str) -> str:
    if energy_status != "accepted":
        return energy_status
    if pure_status != PURE_STATUS_ACCEPTED:
        return pure_status
    return "accepted"


def scalar_seed_mean(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def scalar_seed_stderr(values: list[float]) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))
