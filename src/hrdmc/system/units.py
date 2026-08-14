from __future__ import annotations

HO_TRAP_OMEGA = 1.0
def harmonic_oscillator_unit_metadata() -> dict[str, float | str]:
    return {
        "coordinate": "q = x/a_ho",
        "length_unit": "a_ho = sqrt(hbar/(m*Omega))",
        "energy_coordinate": "E_tilde = E/(hbar*Omega)",
        "energy_unit": "hbar*Omega",
        "time_unit": "1/Omega",
        "report_energy_unit": "hbar*Omega",
    }
