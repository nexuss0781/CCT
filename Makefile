SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
UV ?= uv
MATURIN := $(VENV)/bin/maturin

.PHONY: venv install install-native test test-stage1 lint typecheck benchmark-smoke report stage1-gate ci ci-stage1 clean

venv:
	$(UV) venv --clear --python 3.12 $(VENV)

install: venv
	$(UV) pip install --python $(PYTHON) -e 'causa_py[dev]'
	$(MAKE) install-native

install-native:
	$(MATURIN) develop --release --manifest-path causa_core/Cargo.toml

test:
	$(PYTHON) -m pytest -q

test-stage1:
	$(PYTHON) -m pytest -q tests/test_stage1_numerical_engine.py tests/test_stage0_baseline.py causa_py/tests

lint:
	$(PYTHON) -m compileall -q causa_py scripts
	cargo fmt --manifest-path causa_core/Cargo.toml --all -- --check


typecheck:
	$(PYTHON) -m compileall -q causa_py scripts

benchmark-smoke:
	$(PYTHON) scripts/stage0_harness.py --mode smoke --output artifacts/stage-0/smoke

report:
	$(PYTHON) scripts/stage0_harness.py --mode gate --output artifacts/stage-0/gate

stage1-gate:
	$(PYTHON) scripts/stage1_harness.py --output artifacts/stage-1/gate

ci: install lint test benchmark-smoke report

ci-stage1: install lint test-stage1 stage1-gate

clean:
	rm -rf $(VENV) .pytest_cache artifacts/stage-0 artifacts/stage-1
	cargo clean --manifest-path causa_core/Cargo.toml
