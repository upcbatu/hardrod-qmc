# Proposal

## Title

**Cost-Accuracy Benchmarking of Diffusion Monte Carlo Estimators in One-Dimensional Hard-Rod Bose Gases**

## Abstract

This project proposes a computational thesis on Quantum Monte Carlo estimator performance for the one-dimensional hard-rod Bose gas. The physical system is not chosen as a source of unknown phase behavior, but as a controlled benchmark with well-established reference results for the equation of state and for correlation observables such as the pair distribution function $g(r)$ and the static structure factor $S(k)$ [1]. The main goal is to quantify, for physically relevant observables, the trade-off between bias, variance, and computational cost across four estimator families: a VMC baseline, the mixed DMC estimator, the extrapolated estimator, and the pure estimator obtained through forward walking [2,3]. The intended outcome is an observable- and density-dependent recommendation map indicating when inexpensive estimators are already sufficient and when the additional cost of pure estimation is justified.

## 1. Motivation

The one-dimensional hard-rod Bose gas provides a convenient setting for a methodological study. On the one hand, it is a physically meaningful many-body model with direct connections to strongly correlated quasi-one-dimensional bosonic systems. On the other hand, it is sufficiently controlled to separate numerical estimator effects from genuine physical uncertainty. The hard-rod benchmark is particularly suitable because the main static quantities of interest are already understood in the literature, which makes it possible to evaluate numerical bias in a disciplined way [1,4].

From the Quantum Monte Carlo perspective, the relevant methodological issue is not whether Diffusion Monte Carlo can in principle project to the ground state, but how one should estimate observables once sampling has been obtained. For observables that do not commute with the Hamiltonian, the natural mixed estimator is generally biased by the trial wavefunction. Extrapolated estimators can reduce part of that bias at small additional cost, while pure estimators obtained through forward walking are expected to be less biased but statistically more expensive [2,3]. The central question of the thesis is therefore practical: for a given observable and density regime, which estimator provides the best compromise between accuracy and cost?

This question is not expected to have the same answer for all observables. In particular, the comparison is likely to be most informative for structural observables such as $g(r)$ and especially $S(k)$, where trial-state bias can remain visible, whereas the energy per particle is expected to be less demanding as a benchmark quantity. The estimator differences may also depend strongly on density regime, trial-wavefunction quality, and the possible inclusion of weak inhomogeneity. For this reason, the thesis is framed not as a search for a single universally best estimator, but as a regime-dependent comparison.

## 2. Objective

The objective of the thesis is to use the one-dimensional hard-rod Bose gas as a controlled benchmark for comparing Quantum Monte Carlo estimator families.

More specifically, the project will:

1. implement and validate the hard-rod benchmark geometry and its reference energy;
2. compute static observables such as $E/N$, $g(r)$, and $S(k)$;
3. compare VMC, mixed DMC, extrapolated, and pure estimators;
4. quantify the bias-variance-cost trade-off as a function of density and observable;
5. produce a recommendation map for estimator choice in this benchmark.

## 3. Physical and Methodological Scope

The physical system is the one-dimensional hard-rod Bose gas on a periodic ring of length $L$, with rod length $a$, particle number $N$, and density $\rho = N/L$. In the thermodynamic limit, the reference hard-rod energy per particle is

```text
E/N = pi^2 rho^2 / [3 (1 - rho a)^2]
```

in the units used by the present repository, namely $\hbar^2/(2m)=1$ [1].

The main observables of interest are:

- the energy per particle $E/N$;
- the pair distribution function $g(r)$;
- the static structure factor

```text
S(k) = (1/N) < |sum_j exp(i k x_j)|^2 > .
```

The methodological comparison will focus on the following estimator families:

```text
O_VMC   = <Psi_T | O | Psi_T> / <Psi_T | Psi_T>
O_mixed = <Psi_T | O | Psi_0> / <Psi_T | Psi_0>
O_ext   = 2 O_mixed - O_VMC
O_pure  = <Psi_0 | O | Psi_0> / <Psi_0 | Psi_0> .
```

Here $\Psi_T$ denotes the trial wavefunction and $\Psi_0$ the projected ground state. The pure estimator will be approached numerically through forward walking within the DMC workflow [2,3].

## 4. Methodology

The project is organized in five layers. The `systems/` module defines the hard-rod benchmark and its reference quantities. The `wavefunctions/` module defines the trial state $\Psi_T$. The `monte_carlo/` module contains the VMC and DMC workflows. The `estimators/` module computes observables from sampled coordinates, independently of the sampling method. The `analysis/` module performs blocking analysis, bias and variance estimation, and cost scoring.

The computational workflow is:

```text
choose benchmark parameters
-> choose trial wavefunction
-> generate samples with VMC or DMC
-> compute observables
-> construct estimator families
-> compare bias, variance, MSE, and runtime
```

The current repository already contains an initial VMC workflow, hard-rod geometry utilities, and implementations of $g(r)$ and $S(k)$. The DMC side is already separated at the architectural level so that the estimator comparison can be extended to mixed and pure estimators without changing the observable layer.

## 5. Comparison Criteria

For each observable and estimator family, the analysis will track the following quantities:

```text
bias     = mean(O_hat) - O_ref
variance = Var(O_hat)
MSE      = bias^2 + variance
```

In addition, computational cost will be measured through elapsed runtime under fixed environment conditions. A practical cost metric will be defined by combining error and runtime, for example through

```text
cost_score = MSE * runtime .
```

The aim is not only to identify the least biased estimator, but to determine which estimator is preferable once the statistical and computational costs are considered together.

## 6. Expected Contribution

The expected contribution is methodological rather than phenomenological. The thesis does not aim to introduce a new hard-rod phase diagram. Instead, it aims to provide a clear estimator benchmark for a controlled many-body system. The final outcome should specify, for each observable and density regime, whether the mixed estimator is sufficient, whether the extrapolated estimator provides the best practical compromise, or whether pure estimation through forward walking is necessary.

If time permits, the same framework may be extended to a weakly inhomogeneous setting, such as a shallow periodic external potential, in order to test whether estimator bias becomes more pronounced away from the uniform benchmark.

## 7. Work Plan

The work plan is divided into four stages.

### Stage I. Uniform hard-rod validation

The first stage is the validation of the benchmark geometry and the basic observables. This includes the reference energy, the exclusion structure of $g(r)$, and the qualitative density dependence of $S(k)$.

### Stage II. Mixed and extrapolated estimator study

The second stage is the introduction of the DMC workflow for mixed estimators, followed by the construction of extrapolated estimates from the combined VMC and DMC data.

### Stage III. Pure-estimator study

The third stage is the implementation of forward walking and the study of the dependence of pure-estimator quality on the forward-walking length, variance growth, and computational cost.

### Stage IV. Final comparison

The final stage is the construction of a cost-accuracy comparison across density regimes and observables, leading to a supervisor-facing summary of recommended estimator choices.

## References

[1] F. Mazzanti, G. E. Astrakharchik, J. Boronat, and J. Casulleras, "Ground-State Properties of a One-Dimensional System of Hard Rods," *Physical Review Letters* **100**, 020401 (2008). DOI: 10.1103/PhysRevLett.100.020401.

[2] J. Boronat and J. Casulleras, "Unbiased estimators in quantum Monte Carlo methods: Application to liquid helium," *Physical Review B* **52**, 3654-3661 (1995). DOI: 10.1103/PhysRevB.52.3654.

[3] A. Sarsa, J. Boronat, and J. Casulleras, "Quadratic diffusion Monte Carlo and pure estimators for atoms," *The Journal of Chemical Physics* **116**, 5956-5962 (2002). DOI: 10.1063/1.1446847.

[4] G. E. Astrakharchik, J. Boronat, J. Casulleras, and S. Giorgini, "Beyond the Tonks-Girardeau Gas: Strongly Correlated Regime in Quasi-One-Dimensional Bose Gases," *Physical Review Letters* **95**, 190407 (2005). DOI: 10.1103/PhysRevLett.95.190407.
