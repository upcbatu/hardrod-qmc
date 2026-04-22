PYTHON ?= python3

.PHONY: test smoke clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) experiments/00_smoke_vmc.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
