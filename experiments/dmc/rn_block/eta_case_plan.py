from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.io.artifacts import ensure_dir


@dataclass(frozen=True)
class EtaCase:
    n_particles: int
    rod_length: float
    target_eta_peak: float
    omega: float
    mu_peak: float
    scaled_particle_integral: float
    integration_backend: str

    @property
    def case_id(self) -> str:
        return (
            f"N{self.n_particles}_a{self.rod_length:g}_omega"
            f"{format_case_float(self.omega)}"
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "case_id": self.case_id,
            "n_particles": self.n_particles,
            "rod_length": self.rod_length,
            "target_eta_peak": self.target_eta_peak,
            "omega": self.omega,
            "mu_peak": self.mu_peak,
            "scaled_particle_integral": self.scaled_particle_integral,
            "integration_backend": self.integration_backend,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trapped hard-rod RN-DMC case ids from target LDA peak "
            "packing fractions eta_peak=a*n(0)."
        )
    )
    parser.add_argument("--n-values", default="4,8,12,16")
    parser.add_argument("--rod-length", type=float, default=0.5)
    parser.add_argument("--eta-values", default="0.08,0.15,0.25,0.35,0.45")
    parser.add_argument("--quad-order", type=int, default=800)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_values = parse_ints(args.n_values)
    eta_values = parse_floats(args.eta_values)
    cases = build_cases(
        n_values=n_values,
        rod_length=args.rod_length,
        eta_values=eta_values,
        quad_order=args.quad_order,
        progress=args.progress,
    )
    payload = {
        "status": "completed",
        "claim_boundary": (
            "LDA planning grid only; generated omega values are target-case "
            "design inputs, not DMC results."
        ),
        "n_values": n_values,
        "rod_length": args.rod_length,
        "eta_values": eta_values,
        "case_count": len(cases),
        "cases_csv": ",".join(case.case_id for case in cases),
        "cases": [case.to_dict() for case in cases],
    }
    if not args.no_write:
        output_dir = args.output_dir or artifact_dir(
            repo_root_from(Path(__file__)),
            ArtifactRoute("dmc", "rn_block", "eta_case_plan"),
        )
        write_outputs(output_dir, payload)
    print(json.dumps(payload, indent=2))


def build_cases(
    *,
    n_values: list[int],
    rod_length: float,
    eta_values: list[float],
    quad_order: int,
    progress: bool,
) -> list[EtaCase]:
    if rod_length <= 0.0:
        raise ValueError("rod_length must be positive for eta_peak planning")
    pairs = [(n, eta) for n in n_values for eta in eta_values]
    iterator = progress_iter(pairs, enabled=progress, label="eta cases")
    cases = []
    for n_particles, eta in iterator:
        scaled_integral, mu_peak, backend = scaled_particle_integral(
            eta,
            rod_length,
            quad_order=quad_order,
        )
        cases.append(
            EtaCase(
                n_particles=n_particles,
                rod_length=rod_length,
                target_eta_peak=eta,
                omega=float(scaled_integral / n_particles),
                mu_peak=mu_peak,
                scaled_particle_integral=scaled_integral,
                integration_backend=backend,
            )
        )
    return cases


def scaled_particle_integral(
    eta_peak: float,
    rod_length: float,
    *,
    quad_order: int,
) -> tuple[float, float, str]:
    if not 0.0 < eta_peak < 1.0:
        raise ValueError("eta_peak must satisfy 0 < eta_peak < 1")
    mu_peak = hard_rod_mu_eta(eta_peak, rod_length)

    def theta_integrand(theta: float) -> float:
        eta = eta_peak * np.sin(theta) ** 2
        if eta <= 0.0:
            return 0.0
        deta = 2.0 * eta_peak * np.sin(theta) * np.cos(theta)
        delta_mu = mu_peak - hard_rod_mu_eta(float(eta), rod_length)
        if delta_mu <= 0.0:
            return endpoint_integrand(eta_peak, rod_length)
        rho = eta / rod_length
        return float(2.0 * rho * dmu_deta(float(eta), rod_length) * deta / np.sqrt(2.0 * delta_mu))

    scipy_quad = load_scipy_quad()
    if scipy_quad is not None:
        value = float(
            scipy_quad(theta_integrand, 0.0, 0.5 * np.pi, epsabs=1e-11, epsrel=1e-11)[0]
        )
        return value, mu_peak, "scipy.quad"
    return gauss_legendre_integral(theta_integrand, order=quad_order), mu_peak, "numpy.legendre"


def hard_rod_mu_eta(eta: float, rod_length: float) -> float:
    return float(
        np.pi**2
        * eta**2
        * (3.0 - eta)
        / (3.0 * rod_length**2 * (1.0 - eta) ** 3)
    )


def dmu_deta(eta: float, rod_length: float) -> float:
    return float(2.0 * np.pi**2 * eta / (rod_length**2 * (1.0 - eta) ** 4))


def endpoint_integrand(eta_peak: float, rod_length: float) -> float:
    return float(
        4.0
        * eta_peak**1.5
        * np.sqrt(0.5 * dmu_deta(eta_peak, rod_length))
        / rod_length
    )


def gauss_legendre_integral(
    function: Callable[[float], float],
    *,
    order: int,
) -> float:
    if order < 16:
        raise ValueError("quad_order must be at least 16")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    low = 0.0
    high = 0.5 * np.pi
    mapped = 0.5 * (high - low) * nodes + 0.5 * (high + low)
    values = np.asarray([function(float(theta)) for theta in mapped], dtype=float)
    return float(0.5 * (high - low) * np.dot(weights, values))


def load_scipy_quad():
    try:
        integrate = importlib.import_module("scipy.integrate")
    except ModuleNotFoundError:
        return None
    return integrate.quad


def write_outputs(output_dir: Path, payload: dict) -> None:
    out = ensure_dir(output_dir)
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = [
        "case_id",
        "n_particles",
        "rod_length",
        "target_eta_peak",
        "omega",
        "mu_peak",
        "scaled_particle_integral",
        "integration_backend",
    ]
    with (out / "case_table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["cases"])


def parse_ints(value: str) -> list[int]:
    out = [int(item) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("at least one integer is required")
    return out


def parse_floats(value: str) -> list[float]:
    out = [float(item) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("at least one float is required")
    return out


def format_case_float(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")


def progress_iter(items: list[tuple[int, float]], *, enabled: bool, label: str):
    if not enabled:
        return items
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return items
    return tqdm(items, desc=label, unit="case")


if __name__ == "__main__":
    main()
