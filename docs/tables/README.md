# Generated DMC Tables

Current commands write CSV tables beside their JSON summaries under `results/`.
This directory intentionally contains no frozen production rows: a table used
in the thesis must remain traceable to its run manifest, configuration
fingerprint, and source summaries.

Use the commands under `experiments/dmc/local/` to generate stationarity,
forward-walking, density, and energy-response tables in harmonic-oscillator
units. Trapped case identifiers use `N*_A*`, where \(A=a/a_{\rm ho}\).
