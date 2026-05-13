# Numerics

Owner: optional numerical backend plumbing.

This package centralizes optional acceleration availability and backend labels.
It does not own guide formulas, system transition formulas, Monte Carlo state
updates, or estimator semantics.

Formula owners keep their hot array kernels in their own packages and import
backend helpers from here.
