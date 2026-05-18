from __future__ import annotations

import math
from dataclasses import dataclass


HO_TRAP_OMEGA_IN_REPO_UNITS = math.sqrt(2.0)
REPO_ENERGY_TO_HBAR_OMEGA = 0.5
HBAR_OMEGA_TO_REPO_ENERGY = 2.0


@dataclass(frozen=True)
class HarmonicOscillatorUnitMetadata:
    """Unit convention used by trapped hard-rod production cases.

    The code Hamiltonian uses the kinetic convention hbar^2/(2m)=1, so a
    harmonic-oscillator length unit a_ho is represented by omega_code=sqrt(2).
    Energies emitted by the engine are therefore in hbar*Omega/2 units. Divide
    by two to report energies in the usual hbar*Omega harmonic-oscillator unit.
    """

    length_unit: str = "a_ho"
    energy_unit: str = "hbar*Omega/2"
    report_energy_unit: str = "hbar*Omega"
    trap_omega_code: float = HO_TRAP_OMEGA_IN_REPO_UNITS
    repo_energy_to_hbar_omega: float = REPO_ENERGY_TO_HBAR_OMEGA


def hbar_omega_energy_from_repo_energy(value: float) -> float:
    return float(value * REPO_ENERGY_TO_HBAR_OMEGA)


def repo_energy_from_hbar_omega_energy(value: float) -> float:
    return float(value * HBAR_OMEGA_TO_REPO_ENERGY)


def harmonic_oscillator_unit_metadata() -> dict[str, float | str]:
    units = HarmonicOscillatorUnitMetadata()
    return {
        "length_unit": units.length_unit,
        "energy_unit": units.energy_unit,
        "report_energy_unit": units.report_energy_unit,
        "trap_omega_code": units.trap_omega_code,
        "repo_energy_to_hbar_omega": units.repo_energy_to_hbar_omega,
    }
