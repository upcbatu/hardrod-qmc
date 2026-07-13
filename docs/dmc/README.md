# DMC Notes

This folder documents the trapped hard-rod DMC implementation. Local
importance-sampled DMC is the default method. A collective
Radon-Nikodym-corrected move is available as an optional scheduled extension.

## Trapped Unit Convention

Production trapped cases are parameterized in harmonic-oscillator length units:

```text
N8_A0.2
```

means \(N=8\) and \(A=a/a_{\mathrm{ho}}=0.2\). The physical trap frequency is
absorbed into \(A\) after this nondimensionalization.
The engine stores dimensionless variables \(q=x/a_{\mathrm{ho}}\) and
\(\widetilde E=E/(\hbar\Omega)\), so the default trapped Hamiltonian in code
variables is \(-\frac12\nabla_q^2+\frac12 q^2+\widetilde V_{\rm HR}\).
Reported energies are in units of \(\hbar\Omega\).

## Contact-guide calibration

The contact-corrected reduced-TG guide has a fail-closed calibration path:

1. `optimize_contact_guide.py` produces a correlated-sampling parameter
   candidate; it is not accepted directly by a DMC production runner.
2. `guide_mala_diagnostic.py` tests that fixed candidate without branching or
   population resampling, once from a compact start and once from an expanded
   start with a different seed.
3. `validate_contact_guide.py` verifies both manifests and compares tail
   geometry, local-energy, proposal, and mobility diagnostics.
4. Benchmark, stationarity, and Hellmann-Feynman runners accept the contact
   guide only through that validated artifact and bind its summary and manifest
   hashes into their own run configuration.

These checks establish branching-free guide-squared MALA stability under the
recorded calibration conditions. They do not establish DMC branching,
population, timestep, or production-seed convergence; those remain separate
short-run and production requirements.

- [method.md](method.md): local DMC flow, optional collective move, observables,
  and numerical checks.
- [transport_contract.md](transport_contract.md): DMC event-stream contract for
  transported auxiliary forward-walking estimators.

Formula ownership and literature references remain in
[../03_EQUATION_SOURCE_MAP.md](../03_EQUATION_SOURCE_MAP.md).
