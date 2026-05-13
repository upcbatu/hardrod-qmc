# Theory

Owner: analytic and semi-analytic reference predictions.

This package owns homogeneous hard-rod EOS utilities, chemical-potential
inversion, LDA normalization, LDA density/energy predictions, exact analytic
anchors such as the trapped Tonks-Girardeau limit, and deterministic low-N
references such as the N=2 trapped finite-a relative-coordinate solve.

Reduced-coordinate geometry identities such as `L' = L - N a` are supplied by
`systems/`; theory uses them but does not own geometry.

It does not compare predictions against sampled observables.
