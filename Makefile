PYTHON ?= python3

.PHONY: check lint typecheck validate-ring validate-ring-grid validate-dmc-exact validate-dmc-trapped-stationarity clean

check: lint typecheck

lint:
	$(PYTHON) -m ruff check

typecheck:
	PYTHONPATH=src $(PYTHON) -m pyright

validate-ring:
	PYTHONPATH=src $(PYTHON) experiments/anchors/homogeneous_ring.py

validate-ring-grid:
	PYTHONPATH=src $(PYTHON) experiments/anchors/homogeneous_ring_exact_grid.py

validate-dmc-exact:
	PYTHONPATH=src $(PYTHON) experiments/anchors/exact_tg_trap.py

validate-dmc-trapped-stationarity:
	PYTHONPATH=src $(PYTHON) experiments/dmc/local/trapped_stationarity_grid.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
