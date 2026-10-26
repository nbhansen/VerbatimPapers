.PHONY: test test-unit test-infra test-integration lint format typecheck check ci dev-setup run clean

test:
	uv run python -m pytest

test-unit:
	uv run python -m pytest tests/unit/

test-infra:
	uv run python -m pytest tests/infrastructure/

test-integration:
	uv run python -m pytest tests/integration/

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

typecheck:
	uv run mypy .

check: ci  ## Run all checks (lint, format, typecheck, tests)

ci:  ## Exactly the steps CI runs, so a green check means a green CI
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .
	uv run python -m pytest tests/unit/
	uv run python -m pytest tests/infrastructure/
	uv run python -m pytest tests/integration/

dev-setup:  ## Set up development environment
	uv sync --extra dev
	uv run pre-commit install

run:
	uv run python app.py

clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf .ruff_cache
