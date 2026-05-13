from __future__ import annotations

import numpy as np

GO_ENERGY_STATIONARITY = {"GO", "WARNING_SPREAD_ONLY"}


def energy_claim_status(stationarity: dict) -> str:
    if not bool(stationarity.get("density_accounting_clean", False)):
        return "ENERGY_DENSITY_ACCOUNTING_NO_GO"
    if not bool(stationarity.get("valid_finite_clean", False)):
        return "ENERGY_HYGIENE_NO_GO"
    if stationarity.get("rn_weight_status") != "RN_WEIGHT_GO":
        return "ENERGY_RN_WEIGHT_NO_GO"
    if stationarity.get("stationarity_energy") not in GO_ENERGY_STATIONARITY:
        return "ENERGY_STATIONARITY_NO_GO"
    return "ENERGY_GO"


def pure_fw_claim_status(pure_summary: dict) -> str:
    if pure_summary.get("status") != "PURE_WALKING_GO":
        return str(pure_summary.get("status", "PURE_WALKING_NO_GO"))
    return "PURE_FW_GO"


def benchmark_packet_status(*, energy_status: str, pure_status: str) -> str:
    if energy_status != "ENERGY_GO":
        return energy_status
    if pure_status != "PURE_FW_GO":
        return pure_status
    return "BENCHMARK_PACKET_GO"


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
