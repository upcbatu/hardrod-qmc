# Workflows

Owner: orchestration inside the library.

Workflows assemble systems, wavefunctions, Monte Carlo engines, estimators,
analysis gates, and IO into reusable run plans. They may choose which owner
objects to connect, but they do not own physics formulas, guide formulas,
Monte Carlo transition rules, or gate definitions.

Experiment scripts should be thin wrappers around workflows.
