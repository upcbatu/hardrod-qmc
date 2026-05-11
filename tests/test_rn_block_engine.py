from __future__ import annotations

import numpy as np

from hrdmc.monte_carlo.dmc.rn_block import (
    RNBlockDMCConfig,
    RNBlockLocalStepResult,
    run_rn_block_dmc,
    run_rn_block_dmc_streaming,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions import DMCGuide


class QuadraticGuide:
    def log_value(self, positions: np.ndarray) -> float:
        return -0.5 * float(np.sum(np.asarray(positions, dtype=float) ** 2))

    def grad_log_value(self, positions: np.ndarray) -> np.ndarray:
        return -np.asarray(positions, dtype=float)

    def lap_log_value(self, positions: np.ndarray) -> np.ndarray:
        return -np.ones_like(np.asarray(positions, dtype=float))

    def local_energy(self, positions: np.ndarray) -> float:
        return float(np.sum(np.asarray(positions, dtype=float) ** 2))

    def is_valid(self, positions: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(positions)))


class ShiftProposalKernel:
    def __init__(self, shift: float = 0.1) -> None:
        self.shift = shift
        self.sample_call_count = 0
        self.log_density_call_count = 0

    def sample(
        self,
        rng: np.random.Generator,
        x_old: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        del rng, tau
        self.sample_call_count += 1
        return np.asarray(x_old, dtype=float) + self.shift

    def log_density(self, x_old: np.ndarray, x_new: np.ndarray, tau: float) -> np.ndarray:
        del tau
        self.log_density_call_count += 1
        x_old = np.asarray(x_old, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        return np.zeros(x_old.shape[0] if x_old.ndim == 2 else 1, dtype=float)


class ConstantTargetKernel:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.log_density_call_count = 0

    def log_density(self, x_old: np.ndarray, x_new: np.ndarray, tau: float) -> np.ndarray:
        del x_new, tau
        self.log_density_call_count += 1
        x_old = np.asarray(x_old, dtype=float)
        return np.full(x_old.shape[0] if x_old.ndim == 2 else 1, self.value, dtype=float)


def test_rn_block_loop_accepts_injected_local_step_without_rn_events() -> None:
    guide = QuadraticGuide()
    system = OpenLineHardRodSystem(n_particles=2, rod_length=0.0)
    target = ConstantTargetKernel()
    proposal = ShiftProposalKernel()
    initial = np.array([[0.0, 1.0], [1.0, 2.0]])

    def deterministic_step(
        rng: np.random.Generator,
        positions: np.ndarray,
        guide: DMCGuide,
        dt: float,
        local_energies: np.ndarray,
    ) -> RNBlockLocalStepResult:
        del rng, guide, dt, local_energies
        next_positions = positions + 1.0
        next_energies = np.sum(next_positions * next_positions, axis=1)
        return RNBlockLocalStepResult(
            positions=next_positions,
            local_energies=next_energies,
            killed=np.zeros(positions.shape[0], dtype=bool),
        )

    result = run_rn_block_dmc(
        initial_walkers=initial,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        config=RNBlockDMCConfig(rn_cadence_tau=100.0),
        rng=np.random.default_rng(1),
        dt=0.1,
        burn_in_steps=1,
        production_steps=2,
        store_every=1,
        local_step=deterministic_step,
    )

    assert result.metadata["rn_event_count"] == 0
    assert result.metadata["local_step_count"] == 3
    assert result.metadata["guide_batch_backend"] == "scalar"
    assert result.snapshots.shape == (4, 2)
    np.testing.assert_allclose(result.snapshots[:2], initial + 2.0)
    np.testing.assert_allclose(result.snapshots[2:], initial + 3.0)


def test_rn_block_loop_uses_external_target_and_proposal_kernels_for_rn_step() -> None:
    guide = QuadraticGuide()
    system = OpenLineHardRodSystem(n_particles=2, rod_length=0.0)
    target = ConstantTargetKernel(value=0.25)
    proposal = ShiftProposalKernel(shift=0.1)
    initial = np.array([[0.0, 1.0], [1.0, 2.0]])

    result = run_rn_block_dmc(
        initial_walkers=initial,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        config=RNBlockDMCConfig(
            tau_block=0.01,
            rn_cadence_tau=0.01,
            component_log_scales=(0.0,),
            component_probabilities=(1.0,),
        ),
        rng=np.random.default_rng(2),
        dt=0.01,
        burn_in_steps=0,
        production_steps=1,
        store_every=1,
        include_guide_ratio=False,
    )

    assert result.metadata["rn_event_count"] == 1
    assert result.metadata["local_step_count"] == 0
    assert target.log_density_call_count == 1
    assert proposal.sample_call_count == 1
    assert proposal.log_density_call_count == 2
    np.testing.assert_allclose(result.snapshots, initial + 0.1)
    np.testing.assert_allclose(result.weights, np.array([0.5, 0.5]))


def test_streaming_summary_matches_raw_result_without_retaining_snapshots() -> None:
    guide = QuadraticGuide()
    system = OpenLineHardRodSystem(n_particles=2, rod_length=0.0)
    target = ConstantTargetKernel()
    proposal = ShiftProposalKernel()
    initial = np.array([[0.0, 1.0], [1.0, 2.0]])
    grid = np.array([2.0, 3.0, 4.0, 5.0])

    def deterministic_step(
        rng: np.random.Generator,
        positions: np.ndarray,
        guide: DMCGuide,
        dt: float,
        local_energies: np.ndarray,
    ) -> RNBlockLocalStepResult:
        del rng, guide, dt, local_energies
        next_positions = positions + 1.0
        next_energies = np.sum(next_positions * next_positions, axis=1)
        return RNBlockLocalStepResult(
            positions=next_positions,
            local_energies=next_energies,
            killed=np.zeros(positions.shape[0], dtype=bool),
        )

    raw = run_rn_block_dmc(
        initial_walkers=initial,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        config=RNBlockDMCConfig(rn_cadence_tau=100.0),
        rng=np.random.default_rng(1),
        dt=0.1,
        burn_in_steps=1,
        production_steps=2,
        store_every=1,
        local_step=deterministic_step,
    )
    streaming = run_rn_block_dmc_streaming(
        initial_walkers=initial,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        density_grid=grid,
        config=RNBlockDMCConfig(rn_cadence_tau=100.0),
        rng=np.random.default_rng(1),
        dt=0.1,
        burn_in_steps=1,
        production_steps=2,
        store_every=1,
        local_step=deterministic_step,
    )
    raw_observables = raw.estimate_weighted_observables(
        valid_mask=np.ones(raw.snapshots.shape[0], dtype=bool),
        grid=grid,
        center=system.center,
        n_particles=system.n_particles,
    )

    assert streaming.metadata["summary_mode"] == "streaming"
    assert streaming.metadata["guide_batch_backend"] == "scalar"
    assert streaming.stored_batch_count == raw.metadata["stored_batch_count"]
    np.testing.assert_allclose(streaming.mixed_energy, raw_observables.mixed_energy)
    np.testing.assert_allclose(streaming.r2_radius, raw_observables.r2_radius)
    np.testing.assert_allclose(streaming.rms_radius, raw_observables.rms_radius)
    np.testing.assert_allclose(streaming.density, raw_observables.density.n_x)
    np.testing.assert_allclose(streaming.density_integral, raw_observables.density_integral)
    assert streaming.trace_times is not None
    assert streaming.mixed_energy_trace is not None
    assert streaming.rms_radius_trace is not None
    assert streaming.ess_fraction_trace is not None
    assert streaming.log_weight_span_trace is not None
    assert streaming.invalid_proposal_fraction_trace is not None
    assert streaming.rn_logk_mean_trace is not None
    assert streaming.trace_times.size == streaming.mixed_energy_trace.size
    assert streaming.trace_times.size == streaming.rms_radius_trace.size


def test_streaming_checkpoint_resume_reconstructs_completed_seed(tmp_path) -> None:
    guide = QuadraticGuide()
    system = OpenLineHardRodSystem(n_particles=2, rod_length=0.0)
    target = ConstantTargetKernel()
    proposal = ShiftProposalKernel()
    initial = np.array([[0.0, 1.0], [1.0, 2.0]])
    grid = np.array([2.0, 3.0, 4.0, 5.0])
    checkpoint = tmp_path / "seed_checkpoint.npz"

    def deterministic_step(
        rng: np.random.Generator,
        positions: np.ndarray,
        guide: DMCGuide,
        dt: float,
        local_energies: np.ndarray,
    ) -> RNBlockLocalStepResult:
        del rng, guide, dt, local_energies
        next_positions = positions + 1.0
        next_energies = np.sum(next_positions * next_positions, axis=1)
        return RNBlockLocalStepResult(
            positions=next_positions,
            local_energies=next_energies,
            killed=np.zeros(positions.shape[0], dtype=bool),
        )

    first = run_rn_block_dmc_streaming(
        initial_walkers=initial,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        density_grid=grid,
        config=RNBlockDMCConfig(rn_cadence_tau=100.0),
        rng=np.random.default_rng(1),
        dt=0.1,
        burn_in_steps=1,
        production_steps=2,
        store_every=1,
        local_step=deterministic_step,
        checkpoint_path=checkpoint,
        checkpoint_every_steps=1,
    )
    resumed = run_rn_block_dmc_streaming(
        initial_walkers=initial + 100.0,
        guide=guide,
        system=system,
        target_kernel=target,
        proposal_kernel=proposal,
        density_grid=grid,
        config=RNBlockDMCConfig(rn_cadence_tau=100.0),
        rng=np.random.default_rng(999),
        dt=0.1,
        burn_in_steps=1,
        production_steps=2,
        store_every=1,
        local_step=deterministic_step,
        checkpoint_path=checkpoint,
        checkpoint_every_steps=1,
        resume=True,
    )

    assert checkpoint.is_file()
    np.testing.assert_allclose(resumed.mixed_energy, first.mixed_energy)
    np.testing.assert_allclose(resumed.r2_radius, first.r2_radius)
    np.testing.assert_allclose(resumed.density, first.density)
    assert resumed.stored_batch_count == first.stored_batch_count
