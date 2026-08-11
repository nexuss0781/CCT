SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
UV ?= uv
MATURIN := $(VENV)/bin/maturin

.PHONY: venv install install-native test lint typecheck benchmark-smoke report ci clean

venv:
	$(UV) venv --clear --python 3.12 $(VENV)

install: venv
	$(UV) pip install --python $(PYTHON) -e 'causa_py[dev]'
	$(MAKE) install-native

install-native:
	$(MATURIN) develop --release --manifest-path causa_core/Cargo.toml

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q causa_py scripts
	cargo fmt --manifest-path causa_core/Cargo.toml --all -- --check


typecheck:
	$(PYTHON) -m compileall -q causa_py scripts

benchmark-smoke:
	$(PYTHON) scripts/stage0_harness.py --mode smoke --output artifacts/stage-0/smoke

report:
	$(PYTHON) scripts/stage0_harness.py --mode gate --output artifacts/stage-0/gate

ci: install lint test benchmark-smoke report

clean:
	rm -rf $(VENV) .pytest_cache artifacts/stage-0
	cargo clean --manifest-path causa_core/Cargo.toml
