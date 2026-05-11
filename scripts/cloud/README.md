# Cloud RN-DMC Runs

These scripts are thin operators for running RN-DMC campaigns on a Linux VM.

## Bootstrap

```bash
bash scripts/cloud/bootstrap_vm.sh
```

This installs system build tools, creates `.venv`, installs `.[dmc,dev]`, and
runs the local checks unless `HRDMC_SKIP_CHECKS=1`.

## Eta Campaign

Default campaign:

```bash
bash scripts/cloud/run_eta_campaign.sh
```

Useful overrides:

```bash
HRDMC_N_VALUES=4,8,16 \
HRDMC_ETA_VALUES=0.05,0.10,0.20,0.30,0.45,0.60 \
HRDMC_SEEDS=1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012 \
HRDMC_PARALLEL_WORKERS=12 \
HRDMC_PRODUCTION_TAU=480 \
bash scripts/cloud/run_eta_campaign.sh
```

High-N discovery should use shorter runs first:

```bash
HRDMC_N_VALUES=32,64 \
HRDMC_ETA_VALUES=0.10,0.30,0.45 \
HRDMC_SEEDS=401,402,403,404 \
HRDMC_PARALLEL_WORKERS=4 \
HRDMC_PRODUCTION_TAU=120 \
bash scripts/cloud/run_eta_campaign.sh
```
