#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HRDMC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${HRDMC_VENV:-$ROOT_DIR/.venv}"
SKIP_APT="${HRDMC_SKIP_APT:-0}"
SKIP_CHECKS="${HRDMC_SKIP_CHECKS:-0}"

cd "$ROOT_DIR"

if [[ "$SKIP_APT" != "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-pip build-essential tmux
fi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel
python -m pip install -e ".[dmc,dev]"

if [[ "$SKIP_CHECKS" != "1" ]]; then
  PYTHONPATH=src python -m ruff check .
  PYTHONPATH=src python -m pyright
  TMPDIR="${TMPDIR:-$ROOT_DIR/runtime_tmp}" PYTHONPATH=src python -m pytest -q
fi

python - <<'PY'
import importlib.util

for name in ("numpy", "numba", "scipy", "tqdm", "matplotlib"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'ok' if spec is not None else 'missing'}")
PY

echo "bootstrap complete: $ROOT_DIR"
