# Proposal

## Title

**Trapped One-Dimensional Hard-Rod Bosons: QMC Benchmarks of Excluded-Volume Local-Density Theory**

## Abstract

This project studies trapped one-dimensional hard-rod bosons as the main thesis problem. The homogeneous hard-rod gas on a periodic ring is used as a controlled validation benchmark for the Quantum Monte Carlo workflow, because its ground-state energy and wavefunction are known through the excluded-volume mapping [1]. Available trapped Bose-gas QMC and hard-sphere/hard-rod wavefunction results provide additional reference anchors for the inhomogeneous problem [2,4]. The central comparison is not the construction of LDA itself, but a benchmark of where an excluded-volume LDA built from the homogeneous hard-rod equation of state succeeds and where it fails. The thesis asks how far the excluded-volume idea can be pushed from the exactly controlled homogeneous system into a spatially varying trapped gas.

## 1. Motivation

One-dimensional hard rods are a useful model for strongly correlated bosons because the hard core introduces a finite excluded length while preserving an analytically controlled homogeneous reference. On a ring, the problem maps to point-like particles living on the reduced length `L' = L - N a`, where `a` is the rod length. This gives an exact benchmark for the homogeneous energy and a physically transparent equation of state [1].

The trapped problem is more relevant as a many-body target because real quasi-one-dimensional gases are normally inhomogeneous. In a trap, translation invariance is lost and the density profile becomes a primary observable. LDA methods for trapped systems are already established, including work that derives density profiles, Thomas-Fermi sizes, and energy relations from perturbative equations of state [3]. The contribution here is therefore not to introduce LDA, but to test the accuracy limits of the excluded-volume LDA in a controlled hard-rod setting.

The scope also should not claim that trapped hard rods are being solved for the first time. There are already strong trapped hard-sphere/hard-rod wavefunction results in the literature [4]. The thesis contribution is a controlled QMC and benchmark-data failure map for excluded-volume LDA, especially where finite particle number, trap-edge behavior, density profiles, and correlation observables can expose the limits of a local approximation.

The former estimator-cost benchmark remains useful as infrastructure, especially for separating VMC and DMC data products and for tracking uncertainty. It is no longer the main thesis objective. The main scientific comparison is now trapped QMC benchmarks versus excluded-volume LDA, with DMC as the target production method and VMC as a baseline and diagnostic stage.

## 2. Objective

The objective is to benchmark the regime of validity of excluded-volume LDA for trapped hard rods, using homogeneous exact results and available trapped wavefunction/QMC results as validation anchors.

More specifically, the project will:

1. validate the homogeneous hard-rod implementation on a ring using exact energy and wavefunction information;
2. include available trapped hard-sphere/hard-rod wavefunction and QMC results as reference context;
3. implement a trapped one-dimensional hard-rod system with open-line geometry and a harmonic external potential;
4. compute trapped benchmark observables with QMC, with DMC as the target production method and VMC as a baseline and diagnostic stage;
5. quantify LDA accuracy and failure regimes across particle number, rod length, trap strength, trap edge, and correlation observables.

## 3. Physical Scope

The validation system is the homogeneous one-dimensional hard-rod Bose gas on a periodic ring of length `L`, rod length `a`, particle number `N`, and density `rho = N/L`.

In the thermodynamic limit, the hard-rod energy per particle in repository units \(\hbar^2/(2m)=1\) is

$$
e_{\mathrm{HR}}(\rho)=\frac{E}{N}
=\frac{\pi^2 \rho^2}{3(1-a\rho)^2}.
$$

The corresponding homogeneous energy density is

$$
\epsilon_{\mathrm{HR}}(\rho)
=\rho e_{\mathrm{HR}}(\rho)
=\frac{\pi^2\rho^3}{3(1-a\rho)^2}.
$$

The local chemical potential used by the LDA is

$$
\mu_{\mathrm{HR}}(\rho)
=\frac{d\epsilon_{\mathrm{HR}}}{d\rho}
=\frac{\pi^2\rho^2(3-a\rho)}{3(1-a\rho)^3}.
$$

For a trapped system with external potential `V_trap(x)`, the LDA density satisfies

$$
\mu_0 = V_{\mathrm{trap}}(x) + \mu_{\mathrm{HR}}\!\left(n_{\mathrm{LDA}}(x)\right)
$$

inside the cloud, with `n_LDA(x)=0` outside the classically allowed region. The global chemical potential `mu_0` is fixed by

$$
\int n_{\mathrm{LDA}}(x)\,dx = N.
$$

The main trapped observables are:

- density profile `n(x)`;
- cloud radius or edge position;
- total energy and potential-energy contribution;
- trap-edge behavior;
- finite-`N` dependence;
- selected correlation observables such as `g(r)`, `g_2(x,x')`, or `S(k)` where they remain numerically meaningful.

## 4. Methodology

The repository remains organized into separate physical, sampling, observable, and analysis layers.

The homogeneous ring workflow is:

```text
choose N, L, a
-> validate hard-rod exclusion and exact energy
-> validate trial wavefunction behavior
-> run VMC checks and DMC checks when available
-> compare sampled quantities with homogeneous references
```

The trapped workflow is:

```text
choose N, a, trap strength
-> generate trapped QMC samples
-> estimate n(x), radius, and energy
-> evaluate the excluded-volume LDA
-> compare benchmark observables with LDA predictions
```

The DMC implementation remains the target production method because the final trapped comparison should target ground-state observables. VMC provides smoke tests, trial-state diagnostics, and baseline data; it should not be presented as final trapped-system validation unless DMC or other benchmark data are unavailable.

The pure-estimator and forward-walking reference [5] is method background if DMC estimator support becomes necessary. It does not define the central thesis contribution.

## 5. Comparison Criteria

For homogeneous validation, the primary criteria are agreement with exact or known reference quantities:

$$
\Delta E_{\mathrm{hom}} = E_{\mathrm{QMC}} - E_{\mathrm{exact}}.
$$

For the trapped system, the primary comparison is between benchmark data and LDA:

$$
\Delta E = E_{\mathrm{benchmark}} - E_{\mathrm{LDA}}.
$$

$$
\Delta n_2 = \int
\left|n_{\mathrm{benchmark}}(x)-n_{\mathrm{LDA}}(x)\right|^2\,dx.
$$

$$
\Delta R = R_{\mathrm{benchmark}} - R_{\mathrm{LDA}}.
$$

The comparison should explicitly separate bulk agreement from trap-edge failure. Finite-`N` dependence is part of the target result, because LDA is expected to improve in larger, smoother systems but can fail for small or moderately sized trapped gases. Correlation observables are also important because an LDA may describe density and energy more reliably than nonlocal quantities.

Uncertainty will be estimated with blocking or repeated seeds where appropriate. Runtime and estimator-family differences may be reported as supporting diagnostics, but they are not the main thesis endpoint.

## 6. Expected Contribution

The contribution is not the hard-rod model, the LDA formalism, or DMC itself. The contribution is a controlled benchmark of how far an excluded-volume LDA built from the homogeneous hard-rod equation of state can reproduce trapped hard-rod observables, using QMC data, DMC when available, and available trapped-system references as validation anchors.

If time permits, the same excluded-volume intuition may be tested beyond hard rods, for example as an approximate route to selected Lieb-Liniger excited-state or correlation-function quantities. This extension is optional and should remain secondary to the trapped hard-rod study.

## 7. Work Plan

### Stage I. Scope and homogeneous validation

Reframe the repository around trapped hard rods, then complete homogeneous ring validation for energy and basic observables.

### Stage II. Trapped-system implementation

Add open-line hard-rod geometry, harmonic trapping, trapped initial states, and non-periodic density-profile estimation.

### Stage III. LDA implementation

Implement the hard-rod equation of state, chemical-potential inversion, trap normalization, and LDA observables.

### Stage IV. QMC benchmarks versus LDA comparison

Run trapped simulations, compute density profiles and energies, and benchmark them against the LDA across the selected parameter grid. The output should be an accuracy and failure map, not a claim that LDA is new.

### Stage V. Optional extension

Only after the trapped hard-rod comparison is complete, test whether the excluded-volume idea gives useful approximations for selected Lieb-Liniger quantities.

## References

[1] F. Mazzanti, G. E. Astrakharchik, J. Boronat, and J. Casulleras, "Ground-State Properties of a One-Dimensional System of Hard Rods," *Physical Review Letters* **100**, 020401 (2008). DOI: 10.1103/PhysRevLett.100.020401.

[2] G. E. Astrakharchik and S. Giorgini, "Quantum Monte Carlo study of the three- to one-dimensional crossover for a trapped Bose gas," *Physical Review A* **66**, 053614 (2002). DOI: 10.1103/PhysRevA.66.053614.

[3] G. E. Astrakharchik, "Local density approximation for a perturbative equation of state," *Physical Review A* **72**, 063620 (2005). DOI: 10.1103/PhysRevA.72.063620.

[4] M. D. Girardeau and G. E. Astrakharchik, "Wave functions of the super-Tonks-Girardeau gas and the trapped one-dimensional hard-sphere Bose gas," *Physical Review A* **81**, 061601(R) (2010). DOI: 10.1103/PhysRevA.81.061601.

[5] J. Boronat and J. Casulleras, "Unbiased estimators in quantum Monte Carlo methods: Application to liquid helium," *Physical Review B* **52**, 3654-3661 (1995). DOI: 10.1103/PhysRevB.52.3654.
