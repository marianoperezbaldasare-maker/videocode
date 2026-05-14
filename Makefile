.PHONY: install dev test lint format clean docs

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e ".[dev]"

dev:
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=claude_vision --cov-report=term-missing

lint:
	ruff check src/
	mypy src/

format:
	ruff format src/

format-check:
	ruff format --check src/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find tests -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docs:
	@echo "Documentation build not yet configured"

all: lint test
