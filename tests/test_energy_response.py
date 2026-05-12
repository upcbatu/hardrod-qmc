from __future__ import annotations

import json

import numpy as np

from hrdmc.estimators import (
    EnergyResponsePoint,
    lambda_from_omega,
    omega_from_lambda,
    omega_ladder_from_relative_lambda_offsets,
    trap_r2_from_energy_response,
)
from hrdmc.workflows.dmc.energy_response import (
    energy_point_gate_from_row,
    reanalyze_trap_r2_energy_response,
)
from hrdmc.workflows.dmc.rn_block import RNCase


def test_lambda_omega_roundtrip() -> None:
    omega = 0.05

    lambda_value = lambda_from_omega(omega)

    assert np.isclose(lambda_value, 0.5 * omega**2)
    assert np.isclose(omega_from_lambda(lambda_value), omega)


def test_relative_lambda_offsets_keep_requested_couplings() -> None:
    omega0 = 0.05
    offsets = (-0.05, 0.0, 0.05)
    lambda0 = lambda_from_omega(omega0)

    omegas = omega_ladder_from_relative_lambda_offsets(omega0, offsets)

    lambdas = np.asarray([lambda_from_omega(omega) for omega in omegas])
    assert np.allclose(lambdas / lambda0 - 1.0, offsets)


def test_trap_r2_energy_response_recovers_quadratic_slope() -> None:
    n_particles = 8
    omega0 = 0.05
    lambda0 = lambda_from_omega(omega0)
    pure_r2 = 132.5
    slope = n_particles * pure_r2
    curvature = 30.0
    offsets = (-0.05, -0.025, 0.0, 0.025, 0.05)
    points = tuple(
        EnergyResponsePoint(
            lambda_value=lambda0 * (1.0 + offset),
            energy=2.5
            + slope * (lambda0 * (1.0 + offset) - lambda0)
            + 0.5 * curvature * (lambda0 * (1.0 + offset) - lambda0) ** 2,
            energy_stderr=1.0e-4,
        )
        for offset in offsets
    )

    result = trap_r2_from_energy_response(
        points,
        n_particles=n_particles,
        omega0=omega0,
        degree=2,
    )

    assert result.fit_response_status == "ENERGY_RESPONSE_GO"
    assert np.isclose(result.pure_r2, pure_r2)
    assert np.isclose(result.paper_rms_radius, np.sqrt(result.pure_r2))


def test_trap_r2_energy_response_rejects_negative_radius() -> None:
    omega0 = 0.05
    lambda0 = lambda_from_omega(omega0)
    points = (
        EnergyResponsePoint(lambda_value=0.95 * lambda0, energy=2.0),
        EnergyResponsePoint(lambda_value=lambda0, energy=1.9),
        EnergyResponsePoint(lambda_value=1.05 * lambda0, energy=1.8),
    )

    result = trap_r2_from_energy_response(points, n_particles=4, omega0=omega0, degree=1)

    assert result.fit_response_status == "ENERGY_RESPONSE_NEGATIVE_R2_NO_GO"
    assert not np.isfinite(result.paper_rms_radius)


def test_offline_reanalysis_fails_closed_without_gate_metadata(tmp_path) -> None:
    base_case = RNCase(n_particles=8, rod_length=0.5, omega=0.05)
    lambda0 = lambda_from_omega(base_case.omega)
    summary = {
        "cases": [
            {
                "case_id": "minus",
                "omega": omega_from_lambda(0.975 * lambda0),
                "mixed_energy": 2.47,
                "mixed_energy_conservative_stderr": 1.0e-3,
            },
            {
                "case_id": "plus",
                "omega": omega_from_lambda(1.025 * lambda0),
                "mixed_energy": 2.53,
                "mixed_energy_conservative_stderr": 1.0e-3,
            },
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    result = reanalyze_trap_r2_energy_response(
        base_case=base_case,
        summary_paths=[path],
        degree=1,
    )

    assert result["fit_response_status"] == "ENERGY_RESPONSE_GO"
    assert result["response_status"] == "ENERGY_RESPONSE_POINT_NO_GO"
    assert not result["paper_grade_eligible"]
    assert {
        row["energy_point_gate"] for row in result["points"]
    } == {"ENERGY_POINT_METADATA_UNAVAILABLE"}


def test_energy_point_gate_requires_rn_weight_go() -> None:
    row = {
        "rn_weight_status": "RN_WEIGHT_WARNING",
        "density_accounting_clean": True,
        "valid_finite_clean": True,
        "blocking_plateau_energy": True,
        "stationarity_energy": "GO",
    }

    assert energy_point_gate_from_row(row) == "ENERGY_POINT_RN_WEIGHT_WARNING"
