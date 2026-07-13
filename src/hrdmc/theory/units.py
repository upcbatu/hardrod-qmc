from __future__ import annotations

import math
from dataclasses import dataclass

HO_TRAP_OMEGA = 1.0


def oscillator_length(*, hbar: float, mass: float, omega: float) -> float:
    """Return the physical harmonic-oscillator length sqrt(hbar / (m Omega))."""
    _validate_positive("hbar", hbar)
    _validate_positive("mass", mass)
    _validate_positive("omega", omega)
    return math.sqrt(hbar / (mass * omega))


def oscillator_energy(*, hbar: float, omega: float) -> float:
    """Return the physical harmonic-oscillator energy scale hbar Omega."""
    _validate_positive("hbar", hbar)
    _validate_positive("omega", omega)
    return hbar * omega


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class HarmonicOscillatorUnits:
    """Physical scales behind the dimensionless trapped calculation.

    The simulation stores dimensionless coordinates and energies,

        q = x / a_ho,
        E_tilde = E / (hbar Omega),

    where a_ho = sqrt(hbar / (m Omega)).  In those variables the trap
    coefficient is one and the one-body oscillator is

        H_tilde = -1/2 d^2/dq^2 + 1/2 q^2.

    The natural-unit shorthand for this representation does not mean that the
    physical constants have been changed; it means every reported coordinate
    and energy has already been divided by the scales above.
    """

    hbar: float
    mass: float
    omega: float

    def __post_init__(self) -> None:
        _validate_positive("hbar", self.hbar)
        _validate_positive("mass", self.mass)
        _validate_positive("omega", self.omega)

    @property
    def length_unit(self) -> float:
        """Physical length represented by one code length unit."""
        return oscillator_length(hbar=self.hbar, mass=self.mass, omega=self.omega)

    @property
    def energy_unit(self) -> float:
        """Physical energy represented by one code energy unit."""
        return oscillator_energy(hbar=self.hbar, omega=self.omega)

    @property
    def time_unit(self) -> float:
        """Physical time represented by one code time unit."""
        return 1.0 / self.omega

    def length_to_code(self, length: float) -> float:
        return float(length) / self.length_unit

    def length_from_code(self, value: float) -> float:
        return float(value) * self.length_unit

    def energy_to_code(self, energy: float) -> float:
        return float(energy) / self.energy_unit

    def energy_from_code(self, value: float) -> float:
        return float(value) * self.energy_unit

    def time_to_code(self, time: float) -> float:
        return float(time) / self.time_unit

    def time_from_code(self, value: float) -> float:
        return float(value) * self.time_unit

    def density_to_code(self, density: float) -> float:
        """Convert a physical 1D density to dimensionless density."""
        return float(density) * self.length_unit

    def density_from_code(self, value: float) -> float:
        """Convert a dimensionless 1D density to physical inverse length."""
        return float(value) / self.length_unit

    def rod_length_to_code(self, rod_length: float) -> float:
        """Convert a physical hard-rod diameter to A = a / a_ho."""
        return self.length_to_code(rod_length)

    def rod_length_from_code(self, value: float) -> float:
        """Convert A = a / a_ho back to a physical hard-rod diameter."""
        return self.length_from_code(value)


@dataclass(frozen=True)
class HarmonicOscillatorUnitMetadata:
    """Unit convention used by trapped hard-rod cases.

    Trapped runs store dimensionless variables:

        q = x / a_ho,
        E_tilde = E / (hbar Omega),
        a_ho = sqrt(hbar / (m Omega)).

    In code variables the one-body oscillator is

        H_tilde = -1/2 d^2/dq^2 + 1/2 q^2

    and case ids use the particle count and the dimensionless hard-rod length
    A = a/a_ho.
    """

    coordinate: str = "q = x/a_ho"
    length_unit: str = "a_ho = sqrt(hbar/(m*Omega))"
    energy_coordinate: str = "E_tilde = E/(hbar*Omega)"
    energy_unit: str = "hbar*Omega"
    time_unit: str = "1/Omega"
    report_energy_unit: str = "hbar*Omega"


def harmonic_oscillator_unit_metadata() -> dict[str, float | str]:
    units = HarmonicOscillatorUnitMetadata()
    return {
        "coordinate": units.coordinate,
        "length_unit": units.length_unit,
        "energy_coordinate": units.energy_coordinate,
        "energy_unit": units.energy_unit,
        "time_unit": units.time_unit,
        "report_energy_unit": units.report_energy_unit,
    }
