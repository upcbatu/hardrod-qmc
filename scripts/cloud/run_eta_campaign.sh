#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HRDMC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${HRDMC_VENV:-$ROOT_DIR/.venv}"

N_VALUES="${HRDMC_N_VALUES:-4,8,16}"
ETA_VALUES="${HRDMC_ETA_VALUES:-0.05,0.10,0.20,0.30,0.45,0.60}"
ROD_LENGTH="${HRDMC_ROD_LENGTH:-0.5}"
SEEDS="${HRDMC_SEEDS:-1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012}"
PARALLEL_WORKERS="${HRDMC_PARALLEL_WORKERS:-12}"
PRODUCTION_TAU="${HRDMC_PRODUCTION_TAU:-480}"
BURN_TAU="${HRDMC_BURN_TAU:-60}"
WALKERS="${HRDMC_WALKERS:-256}"
DT="${HRDMC_DT:-0.00125}"
TAU_BLOCK="${HRDMC_TAU_BLOCK:-0.01}"
RN_CADENCE="${HRDMC_RN_CADENCE:-0.005}"
STORE_EVERY="${HRDMC_STORE_EVERY:-40}"
GRID_EXTENT="${HRDMC_GRID_EXTENT:-20}"
N_BINS="${HRDMC_N_BINS:-240}"
SKIP_PLOTS="${HRDMC_SKIP_PLOTS:-0}"
STAMP="${HRDMC_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${HRDMC_OUTPUT_ROOT:-$ROOT_DIR/results/dmc/rn_block/cloud_eta_${STAMP}}"

cd "$ROOT_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "missing venv: $VENV_DIR"
  echo "run: bash scripts/cloud/bootstrap_vm.sh"
  exit 2
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$OUTPUT_ROOT"

cat > "$OUTPUT_ROOT/run_config.json" <<JSON
{
  "n_values": "$N_VALUES",
  "eta_values": "$ETA_VALUES",
  "rod_length": "$ROD_LENGTH",
  "seeds": "$SEEDS",
  "parallel_workers": "$PARALLEL_WORKERS",
  "production_tau": "$PRODUCTION_TAU",
  "burn_tau": "$BURN_TAU",
  "walkers": "$WALKERS",
  "dt": "$DT",
  "tau_block": "$TAU_BLOCK",
  "rn_cadence": "$RN_CADENCE",
  "store_every": "$STORE_EVERY",
  "grid_extent": "$GRID_EXTENT",
  "n_bins": "$N_BINS",
  "output_root": "$OUTPUT_ROOT"
}
JSON

echo "output: $OUTPUT_ROOT"

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python experiments/dmc/rn_block/eta_case_plan.py \
  --n-values "$N_VALUES" \
  --rod-length "$ROD_LENGTH" \
  --eta-values "$ETA_VALUES" \
  --output-dir "$OUTPUT_ROOT/eta_case_plan" \
  --progress

IFS=',' read -r -a n_items <<< "$N_VALUES"
for n_value in "${n_items[@]}"; do
  n_value="$(echo "$n_value" | xargs)"
  if [[ -z "$n_value" ]]; then
    continue
  fi

  cases="$(
    python - "$OUTPUT_ROOT/eta_case_plan/summary.json" "$n_value" <<'PY'
import json
import sys

summary_path = sys.argv[1]
n_value = int(sys.argv[2])
summary = json.load(open(summary_path))
print(",".join(row["case_id"] for row in summary["cases"] if row["n_particles"] == n_value))
PY
  )"

  if [[ -z "$cases" ]]; then
    echo "no cases for N=$n_value"
    continue
  fi

  run_dir="$OUTPUT_ROOT/stationarity_N${n_value}"
  log_path="$OUTPUT_ROOT/stationarity_N${n_value}.log"
  echo "running N=$n_value cases=$cases"

  extra_args=()
  if [[ "$SKIP_PLOTS" == "1" ]]; then
    extra_args+=(--skip-plots)
  fi

  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python experiments/dmc/rn_block/trapped_stationarity_grid.py \
    --cases "$cases" \
    --dt "$DT" \
    --walkers "$WALKERS" \
    --tau "$TAU_BLOCK" \
    --rn-cadence "$RN_CADENCE" \
    --burn-tau "$BURN_TAU" \
    --production-tau "$PRODUCTION_TAU" \
    --store-every "$STORE_EVERY" \
    --grid-extent "$GRID_EXTENT" \
    --n-bins "$N_BINS" \
    --seeds "$SEEDS" \
    --parallel-workers "$PARALLEL_WORKERS" \
    --output-dir "$run_dir" \
    --progress \
    "${extra_args[@]}" 2>&1 | tee "$log_path"
done

echo "campaign complete: $OUTPUT_ROOT"
