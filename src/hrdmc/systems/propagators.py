from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.open_line import OpenLineHardRodSystem

FloatArray = NDArray[np.float64]


class TargetTransitionKernel(Protocol):
    """System-owned target transition density for DMC/RN weighting."""

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray: ...


class ProposalTransitionKernel(TargetTransitionKernel, Protocol):
    """Transition kernel that can also sample proposals."""

    def sample(
        self,
        rng: np.random.Generator,
        x_old: FloatArray,
        tau: float,
    ) -> FloatArray: ...


def transition_backend(kernel: TargetTransitionKernel) -> str:
    backend = getattr(kernel, "transition_backend", None)
    if isinstance(backend, str):
        return backend
    if callable(backend):
        return str(backend())
    return "numpy"


@dataclass(frozen=True)
class HarmonicMehlerKernel:
    """Normalized one-body harmonic Mehler proposal kernel.

    Repo units use H=-d^2/dx^2 + 0.5*omega^2*x^2, so the oscillator frequency
    entering the normalized Mehler transition is sqrt(2)*omega. This is a
    proposal density, not the raw Hamiltonian heat kernel.
    """

    trap: HarmonicTrap

    def sample(
        self,
        rng: np.random.Generator,
        x_old: FloatArray,
        tau: float,
    ) -> FloatArray:
        x_old = _as_batch(x_old)
        mean, variance = harmonic_transition_moments(x_old, self.trap, tau)
        return mean + math.sqrt(variance) * rng.normal(size=mean.shape)

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = _as_batch(x_old)
        x_new = _as_batch(x_new)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        mean, variance = harmonic_transition_moments(x_old, self.trap, tau)
        return _log_normal_density(x_new, mean, variance)

    @property
    def transition_backend(self) -> str:
        return "numpy"


@dataclass(frozen=True)
class HarmonicOscillatorHeatKernel:
    """Unnormalized one-body harmonic imaginary-time heat kernel.

    This is the Hamiltonian kernel for H=-d^2/dx^2+0.5*omega^2*x^2. It is a
    target kernel for RN/DMC weighting, not a normalized proposal sampler.
    """

    trap: HarmonicTrap

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = _as_batch(x_old)
        x_new = _as_batch(x_new)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        return np.sum(
            harmonic_oscillator_heat_log_matrix(
                x_old,
                x_new,
                self.trap,
                tau,
            )[:, np.arange(x_old.shape[1]), np.arange(x_old.shape[1])],
            axis=1,
        )

    @property
    def transition_backend(self) -> str:
        return "numpy"


@dataclass(frozen=True)
class OrderedHarmonicMehlerKernel:
    """Ordered-sector determinant of normalized harmonic Mehler proposals.

    For zero rod length, the trapped hard-core Bose gas maps to noninteracting
    fermions in one ordered sector. This normalized determinant is useful as a
    proposal/diagnostic density; use OrderedHarmonicOscillatorHeatKernel as the
    raw RN target kernel.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap

    def __post_init__(self) -> None:
        if self.system.rod_length != 0.0:
            raise ValueError("ordered harmonic Mehler kernel requires zero rod length")
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = _as_batch(x_old)
        x_new = _as_batch(x_new)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if x_old.shape[1] != self.system.n_particles:
            raise ValueError("configuration particle count does not match system")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        mean, variance = harmonic_transition_moments(x_old, self.trap, tau)
        diff = x_new[:, np.newaxis, :] - mean[:, :, np.newaxis]
        log_matrix = -0.5 * (math.log(2.0 * math.pi * variance) + (diff * diff) / variance)
        max_log = np.max(log_matrix, axis=(1, 2))
        matrix = np.exp(log_matrix - max_log[:, np.newaxis, np.newaxis])
        sign, logdet = np.linalg.slogdet(matrix)
        out = x_old.shape[1] * max_log + logdet
        return np.where(sign > 0.0, out, -np.inf)

    @property
    def transition_backend(self) -> str:
        return "numpy"


@dataclass(frozen=True)
class OrderedHarmonicOscillatorHeatKernel:
    """Raw ordered-sector harmonic heat kernel for the trapped TG anchor.

    The determinant is the Karlin-McGregor kernel for noninteracting fermions
    before the DMC guide ratio is applied. Use this as RN target with
    include_guide_ratio=True.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap

    def __post_init__(self) -> None:
        if self.system.rod_length != 0.0:
            raise ValueError("ordered harmonic heat kernel requires zero rod length")
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = _as_batch(x_old)
        x_new = _as_batch(x_new)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if x_old.shape[1] != self.system.n_particles:
            raise ValueError("configuration particle count does not match system")
        log_matrix = harmonic_oscillator_heat_log_matrix(x_old, x_new, self.trap, tau)
        max_log = np.max(log_matrix, axis=(1, 2))
        matrix = np.exp(log_matrix - max_log[:, np.newaxis, np.newaxis])
        sign, logdet = np.linalg.slogdet(matrix)
        out = x_old.shape[1] * max_log + logdet
        return np.where(sign > 0.0, out, -np.inf)

    @property
    def transition_backend(self) -> str:
        return "numpy"


def harmonic_oscillator_heat_log_matrix(
    x_old: FloatArray,
    x_new: FloatArray,
    trap: HarmonicTrap,
    tau: float,
) -> FloatArray:
    """Return log K_tau(x_old_i, x_new_j) for the raw oscillator heat kernel."""

    x_old = _as_batch(x_old)
    x_new = _as_batch(x_new)
    if x_old.shape != x_new.shape:
        raise ValueError("x_old and x_new must have matching shapes")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    q = trap.omega / math.sqrt(2.0)
    two_q_tau = 2.0 * q * tau
    sinh = math.sinh(two_q_tau)
    cosh = math.cosh(two_q_tau)
    if sinh <= 0.0 or not math.isfinite(sinh):
        raise ValueError("invalid harmonic heat-kernel time")
    old = x_old - trap.center
    new = x_new - trap.center
    old2 = old[:, :, np.newaxis] * old[:, :, np.newaxis]
    new2 = new[:, np.newaxis, :] * new[:, np.newaxis, :]
    cross = old[:, :, np.newaxis] * new[:, np.newaxis, :]
    log_norm = 0.5 * math.log(q / (2.0 * math.pi * sinh))
    exponent = -q * ((old2 + new2) * cosh - 2.0 * cross) / (2.0 * sinh)
    return log_norm + exponent


def harmonic_transition_moments(
    x_old: FloatArray,
    trap: HarmonicTrap,
    tau: float,
) -> tuple[FloatArray, float]:
    x_old = np.asarray(x_old, dtype=float)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    gamma = math.sqrt(2.0) * trap.omega
    m_gamma = trap.omega / math.sqrt(2.0)
    gamma_tau = gamma * tau
    mean = trap.center + (x_old - trap.center) / math.cosh(gamma_tau)
    variance = math.tanh(gamma_tau) / m_gamma
    return mean, float(variance)


@dataclass(frozen=True)
class OpenHardRodTrapPrimitiveKernel:
    """Primitive hard-rod plus trap target kernel owned by the system layer."""

    system: OpenLineHardRodSystem
    trap: HarmonicTrap

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")

    def log_density(self, x_old: FloatArray, x_new: FloatArray, tau: float) -> FloatArray:
        x_old = _as_batch(x_old)
        x_new = _as_batch(x_new)
        if x_old.shape != x_new.shape:
            raise ValueError("x_old and x_new must have matching shapes")
        if x_old.shape[1] != self.system.n_particles:
            raise ValueError("configuration particle count does not match system")
        log_free = log_free_ordered_hardrod_kernel(
            x_old,
            x_new,
            rod_length=self.system.rod_length,
            tau=tau,
        )
        endpoint_weight = -0.5 * tau * (
            _trap_total_batch(self.trap, x_old) + _trap_total_batch(self.trap, x_new)
        )
        return log_free + endpoint_weight

    @property
    def transition_backend(self) -> str:
        return "numpy"


def log_free_ordered_hardrod_kernel(
    x_old: FloatArray,
    x_new: FloatArray,
    *,
    rod_length: float,
    tau: float,
) -> FloatArray:
    """Karlin-McGregor free ordered kernel in reduced hard-rod coordinates."""

    x_old = _as_batch(x_old)
    x_new = _as_batch(x_new)
    if x_old.shape != x_new.shape:
        raise ValueError("x_old and x_new must have matching shapes")
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    u_old = _to_reduced(x_old, rod_length)
    u_new = _to_reduced(x_new, rod_length)
    n_walkers, n_particles = u_old.shape
    log_norm = -0.5 * math.log(4.0 * math.pi * tau)
    diff = u_new[:, np.newaxis, :] - u_old[:, :, np.newaxis]
    log_matrix = log_norm - (diff * diff) / (4.0 * tau)
    max_log = np.max(log_matrix, axis=(1, 2))
    matrix = np.exp(log_matrix - max_log[:, np.newaxis, np.newaxis])
    sign, logdet = np.linalg.slogdet(matrix)
    out = n_particles * max_log + logdet
    return np.where(sign > 0.0, out, -np.inf)


def _trap_total_batch(trap: HarmonicTrap, x: FloatArray) -> FloatArray:
    return np.sum(trap.values(x), axis=1)


def _log_normal_density(x: FloatArray, mean: FloatArray, variance: float) -> FloatArray:
    if variance <= 0.0 or not math.isfinite(variance):
        raise ValueError("variance must be finite and positive")
    per_dim = -0.5 * (math.log(2.0 * math.pi * variance) + ((x - mean) ** 2) / variance)
    return np.sum(per_dim, axis=-1)


def _to_reduced(x: FloatArray, rod_length: float) -> FloatArray:
    offsets = rod_length * (np.arange(x.shape[-1], dtype=float) - 0.5 * (x.shape[-1] - 1))
    return np.asarray(x, dtype=float) - offsets


def _as_batch(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("configuration array must be one- or two-dimensional")
    return arr
