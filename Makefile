# Makefile — `haku` development convenience targets
#
# Use these instead of remembering long pytest / ruff invocations.
# Targets are intentionally tiny — they exist so Claude can write
# "run `make test`" in the per-step "what to test" checklist
# instead of pasting a multi-line command.
#
# Conventions:
#   - All targets assume /haku/.venv exists. Run `make venv` first
#     on a fresh checkout (or once you've decided what HAKU_HOME points to).
#   - HAKU_HOME defaults to the repo root; override on the command line:
#       make test HAKU_HOME=/home/luis/code/haku
#   - No target deletes user data. `make clean` only touches build artifacts.

HAKU_HOME ?= $(CURDIR)
VENV      := $(HAKU_HOME)/.venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
RUFF      := $(VENV)/bin/ruff
MYPY      := $(VENV)/bin/mypy

.PHONY: help venv install install-dev test lint fmt typecheck check clean

help:
	@echo "haku — developer targets"
	@echo ""
	@echo "  make venv         create /haku/.venv (Python 3.12)"
	@echo "  make install      pip install -r engine/requirements.txt into the venv"
	@echo "  make test         run the full pytest suite"
	@echo "  make lint         run ruff (lint only, no autofix)"
	@echo "  make fmt          run ruff format (rewrites files)"
	@echo "  make typecheck    run mypy on engine/"
	@echo "  make check        lint + typecheck + test (CI-style gate)"
	@echo "  make clean        remove __pycache__, .pytest_cache, .ruff_cache"
	@echo ""
	@echo "Override HAKU_HOME on the command line if needed:"
	@echo "  make test HAKU_HOME=~/code/haku"

venv:
	python3.12 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r engine/requirements.txt
	$(PIP) install -r engine/requirements-dev.txt

test:
	$(PYTEST) engine/ -v

lint:
	$(RUFF) check engine/

fmt:
	$(RUFF) format engine/
	$(RUFF) check --fix engine/

typecheck:
	$(MYPY) engine/

check: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
