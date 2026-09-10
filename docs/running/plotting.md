# Plotting

From the repository root with the [environment activated](../../README.md#installation):

```bash
python experiments/run.py plot \
  --input data/density_profiles/N10_A1_density.csv \
  --reference data/density_profiles/N10_A1_lda.csv \
  --xlim -10 10 \
  --output results/plots/N10_A1_density.png
```

`--input` selects the data file; `--reference` optionally adds a comparison curve.
`--xlim` sets the visible range; omit it for the full grid. Output can be PNG,
PDF or SVG. The command creates the output directory if needed and replaces
an existing image with the same name.

## Plot a new VMC or DMC run

```bash
python experiments/run.py plot \
  --input results/vmc_mala/density.csv \
  --output results/plots/vmc_mala_density.pdf
```

The CSV must already exist. The command recognizes:

| Data | x | y | Optional error |
|---|---|---|---|
| Thesis export | `q_center` | `n_fw` | `n_fw_sem` |
| VMC/DMC output | `q` | `density` | `stderr` |
| LDA | `q` | `n_lda` | None |

With bin edges `q_left,q_right`, the command draws a step plot; otherwise it
connects the points with lines. Bands show plus/minus the supplied error.
No band is drawn if every error entry is blank.
Values are not normalized, smoothed or fitted. Use matching units and
normalization when comparing methods.

## Other CSV columns

Select your file's column names explicitly:

```bash
python experiments/run.py plot \
  --input results/other_method.csv \
  --x position \
  --y rho \
  --yerr rho_error \
  --xlabel "Position" \
  --ylabel "Density" \
  --labels "Other method" \
  --output results/plots/other_method.png
```

Omit `--yerr` if unavailable. x must increase; bins must be valid and contiguous.
For custom reference columns, use `--reference-x` and `--reference-y`.

## Several profiles

Multiple inputs share one axis by default. `--panels` draws one panel per input.
Supply one reference file per input, in the same order:

```bash
python experiments/run.py plot \
  --input \
    data/density_profiles/N10_A0.1_density.csv \
    data/density_profiles/N10_A1_density.csv \
    data/density_profiles/N10_A10_density.csv \
    data/density_profiles/N20_A0.1_density.csv \
    data/density_profiles/N20_A1_density.csv \
    data/density_profiles/N20_A10_density.csv \
  --reference \
    data/density_profiles/N10_A0.1_lda.csv \
    data/density_profiles/N10_A1_lda.csv \
    data/density_profiles/N10_A10_lda.csv \
    data/density_profiles/N20_A0.1_lda.csv \
    data/density_profiles/N20_A1_lda.csv \
    data/density_profiles/N20_A10_lda.csv \
  --panels \
  --output results/plots/finite_rod_density_profiles.pdf
```

`--labels` accepts one quoted label per input; `--title` adds a figure title.
`python experiments/run.py plot --help` lists all options.

See [data definitions](../../data/density_profiles/README.md) for units and
errors. The [original thesis figure](../../results/figures/final_matrix_density_profiles.pdf)
uses the same source data with a different layout.
