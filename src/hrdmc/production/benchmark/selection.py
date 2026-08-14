from __future__ import annotations

import numpy as np

from hrdmc.estimators.forward_walking.results import PURE_STATUS_ACCEPTED


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
