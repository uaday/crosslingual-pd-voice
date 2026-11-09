# Makefile — common project tasks. Run `make help` for the list.
# Most targets are thin wrappers around the pdvoice package.

.PHONY: help env install kernel test lint format clean

help:
	@echo "Targets:"
	@echo "  env       - create .venv and install the requirements"
	@echo "  install   - pip install -e . (editable package)"
	@echo "  kernel    - register the Jupyter kernel 'pdvoice'"
	@echo "  test      - run pytest"
	@echo "  lint      - ruff check"
	@echo "  format    - black + ruff --fix"
	@echo "  clean     - remove caches and build artifacts"

# Windows venvs put the interpreter in Scripts/, Linux/macOS in bin/.
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif

env:
	python -m venv .venv
	$(VENV_PY) -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
	$(VENV_PY) -m pip install -r env/requirements-gpu.txt
	$(VENV_PY) -m pip install -e .

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
