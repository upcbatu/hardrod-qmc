# Systems

Owner: physical geometry and Hamiltonian ingredients.

This package owns ring/open-line geometry, hard-core constraints, boundary
conventions, external potentials, reduced-coordinate geometry identities, and
system-owned transition-kernel interfaces.

System transition classes may call array kernels and report `transition_backend`;
Monte Carlo engines must not branch on numba directly.
`kernels/` owns system-side hot array kernels, such as CDF sampling for
tabulated transition maps.

It does not own the homogeneous equation of state, chemical potential, LDA, or
benchmark error logic.
