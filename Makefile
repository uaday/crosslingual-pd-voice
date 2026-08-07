# Makefile — common project tasks. Run `make help` for the list.
# Most targets are thin wrappers around the pdvoice package / notebooks.

.PHONY: help env install kernel test lint format clean

help:
	@echo "Targets:"
	@echo "  env       - create the conda environment from env/environment.yml"
	@echo "  install   - pip install -e . (editable package)"
	@echo "  kernel    - register the Jupyter kernel 'pdvoice'"
	@echo "  test      - run pytest"
	@echo "  lint      - ruff check"
	@echo "  format    - black + ruff --fix"
	@echo "  clean     - remove caches and build artifacts"

env:
	conda env create -f env/environment.yml

install:
	pip install -e .

kernel:
	python -m ipykernel install --user --name pdvoice --display-name "Python (pdvoice)"

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
