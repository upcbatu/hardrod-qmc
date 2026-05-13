from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from hrdmc.theory import TrappedN2FiniteAReference


@dataclass(frozen=True)
class FiniteAN2ReferenceTolerances:
    energy_abs: float = 0.02
    pure_r2_relative: float = 0.05
    pure_rms_relative: float = 0.03
    pure_density_l2: float = 0.10
    density_accounting_abs: float = 5.0e-3

    def to_payload(self) -> dict[str, float]:
        return {
            "energy_abs": self.energy_abs,
            "pure_r2_relative": self.pure_r2_relative,
            "pure_rms_relative": self.pure_rms_relative,
            "pure_density_l2": self.pure_density_l2,
            "density_accounting_abs": self.density_accounting_abs,
        }


def finite_a_n2_reference_comparison(
    packet: dict[str, Any],
    reference: TrappedN2FiniteAReference,
    *,
    tolerances: FiniteAN2ReferenceTolerances,
) -> dict[str, Any]:
    paper = packet.get("paper_values", {})
    energy = paper.get("energy", {})
    r2 = paper.get("r2", {})
    rms = paper.get("rms", {})
    density = paper.get("density", {})
    bin_edges = _optional_array(density.get("bin_edges"))
    exact_density = _exact_bin_density(reference, bin_edges)
    pure_density = _optional_array(density.get("value"))
    mixed_density = _optional_array(density.get("mixed_diagnostic_value"))
    widths = np.diff(bin_edges) if bin_edges is not None else None

    energy_value = _optional_float(energy.get("value"))
    pure_r2 = _optional_float(r2.get("value"))
    pure_rms = _optional_float(rms.get("value"))
    mixed_r2 = _optional_float(r2.get("mixed_diagnostic"))
    mixed_rms = _optional_float(rms.get("mixed_diagnostic"))
    pure_density_integral = _density_integral(pure_density, widths)
    exact_density_integral = _density_integral(exact_density, widths)
    mixed_density_integral = _density_integral(mixed_density, widths)

    energy_abs_error = _abs_error(energy_value, reference.total_energy)
    pure_r2_relative_error = _relative_error(pure_r2, reference.r2_radius)
    pure_rms_relative_error = _relative_error(pure_rms, reference.rms_radius)
    mixed_r2_relative_error = _relative_error(mixed_r2, reference.r2_radius)
    mixed_rms_relative_error = _relative_error(mixed_rms, reference.rms_radius)
    pure_density_l2 = _density_relative_l2(pure_density, exact_density, widths)
    mixed_density_l2 = _density_relative_l2(mixed_density, exact_density, widths)
    pure_mixed_density_l2 = _density_relative_l2(pure_density, mixed_density, widths)
    density_accounting_abs_error = _abs_error(
        pure_density_integral,
        exact_density_integral,
    )

    gates = {
        "benchmark_packet": packet.get("status") == "BENCHMARK_PACKET_GO",
        "energy_reference": _passes(energy_abs_error, tolerances.energy_abs),
        "pure_r2_reference": _passes(
            pure_r2_relative_error,
            tolerances.pure_r2_relative,
        ),
        "pure_rms_reference": _passes(
            pure_rms_relative_error,
            tolerances.pure_rms_relative,
        ),
        "pure_density_reference": _passes(
            pure_density_l2,
            tolerances.pure_density_l2,
        ),
        "density_accounting": _passes(
            density_accounting_abs_error,
            tolerances.density_accounting_abs,
        ),
    }
    status = finite_a_n2_status(gates)
    return {
        "status": status,
        "gates": {name: "passed" if passed else "failed" for name, passed in gates.items()},
        "tolerances": tolerances.to_payload(),
        "reference": reference.to_metadata(),
        "energy": {
            "dmc": energy_value,
            "reference": reference.total_energy,
            "abs_error": energy_abs_error,
            "status": energy.get("status"),
        },
        "r2": {
            "pure_fw": pure_r2,
            "mixed_diagnostic": mixed_r2,
            "reference": reference.r2_radius,
            "pure_relative_error": pure_r2_relative_error,
            "mixed_relative_error_diagnostic": mixed_r2_relative_error,
            "fw_closer_than_mixed_diagnostic": _left_strictly_smaller(
                pure_r2_relative_error,
                mixed_r2_relative_error,
            ),
            "status": r2.get("status"),
        },
        "rms": {
            "pure_fw": pure_rms,
            "mixed_diagnostic": mixed_rms,
            "reference": reference.rms_radius,
            "pure_relative_error": pure_rms_relative_error,
            "mixed_relative_error_diagnostic": mixed_rms_relative_error,
            "fw_closer_than_mixed_diagnostic": _left_strictly_smaller(
                pure_rms_relative_error,
                mixed_rms_relative_error,
            ),
            "status": rms.get("status"),
        },
        "density": {
            "pure_relative_l2": pure_density_l2,
            "mixed_relative_l2_diagnostic": mixed_density_l2,
            "pure_mixed_relative_l2_diagnostic": pure_mixed_density_l2,
            "fw_closer_than_mixed_diagnostic": _left_strictly_smaller(
                pure_density_l2,
                mixed_density_l2,
            ),
            "pure_integral": pure_density_integral,
            "mixed_integral_diagnostic": mixed_density_integral,
            "reference_integral": exact_density_integral,
            "density_accounting_abs_error": density_accounting_abs_error,
            "bin_edges": bin_edges.tolist() if bin_edges is not None else None,
            "x": _bin_centers(bin_edges).tolist() if bin_edges is not None else None,
            "reference_bin_averaged_n_x": (
                exact_density.tolist() if exact_density is not None else None
            ),
            "pure_fw_n_x": pure_density.tolist() if pure_density is not None else None,
            "mixed_diagnostic_n_x": (
                mixed_density.tolist() if mixed_density is not None else None
            ),
            "status": density.get("status"),
        },
        "claim_boundary": (
            "N=2 finite-a trapped deterministic reference. This validates the "
            "finite-a RN-DMC/FW packet only for N=2 at the declared a, omega, "
            "dt, population, seed, and tolerance settings; it is not an N>2 "
            "exact reference."
        ),
    }


def finite_a_n2_status(gates: dict[str, bool]) -> str:
    if not gates["benchmark_packet"]:
        return "FINITE_A_N2_BENCHMARK_PACKET_NO_GO"
    if not gates["energy_reference"]:
        return "FINITE_A_N2_ENERGY_REFERENCE_NO_GO"
    if not gates["pure_r2_reference"]:
        return "FINITE_A_N2_R2_REFERENCE_NO_GO"
    if not gates["pure_rms_reference"]:
        return "FINITE_A_N2_RMS_REFERENCE_NO_GO"
    if not gates["pure_density_reference"]:
        return "FINITE_A_N2_DENSITY_REFERENCE_NO_GO"
    if not gates["density_accounting"]:
        return "FINITE_A_N2_DENSITY_ACCOUNTING_NO_GO"
    return "FINITE_A_N2_REFERENCE_PACKET_GO"


def _exact_bin_density(
    reference: TrappedN2FiniteAReference,
    bin_edges: np.ndarray | None,
) -> np.ndarray | None:
    if bin_edges is None:
        return None
    return reference.bin_averaged_density(bin_edges)


def _optional_array(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return values


def _optional_float(value: object) -> float | None:
    if value is None or not isinstance(value, Real | str):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _abs_error(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return abs(value - reference)


def _relative_error(value: float | None, reference: float) -> float | None:
    if value is None or reference == 0.0:
        return None
    return abs(value - reference) / abs(reference)


def _density_integral(
    values: np.ndarray | None,
    widths: np.ndarray | None,
) -> float | None:
    if values is None or widths is None or values.shape != widths.shape:
        return None
    return float(np.sum(values * widths))


def _density_relative_l2(
    values: np.ndarray | None,
    reference: np.ndarray | None,
    widths: np.ndarray | None,
) -> float | None:
    if values is None or reference is None or widths is None:
        return None
    if values.shape != reference.shape or values.shape != widths.shape:
        return None
    numerator = float(np.sum((values - reference) ** 2 * widths))
    denominator = float(np.sum(reference * reference * widths))
    if denominator <= 0.0:
        return None
    return float(np.sqrt(numerator / denominator))


def _passes(value: float | None, tolerance: float) -> bool:
    return value is not None and value <= tolerance


def _left_strictly_smaller(left: float | None, right: float | None) -> bool | None:
    if left is None or right is None:
        return None
    return left < right


def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])
