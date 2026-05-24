# MultiMind SDK — common dev tasks.
# Run `make help` to see the full list with descriptions.

.PHONY: help install install-all test test-all test-fast lint format typecheck \
        clean build publish-test publish docs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install core package in editable mode with dev tools
	pip install -e ".[dev]"
	pre-commit install

install-all:  ## Install every extra (rag, agents, compliance, finetune, gateway, …) + dev
	pip install -e ".[all,dev]"
	pre-commit install

test:  ## Run tests, excluding integration + API-key + slow markers
	pytest tests/ -v -m "not integration and not requires_api_key and not slow" --tb=short

test-all:  ## Run the full test suite (markers ignored)
	pytest tests/ -v --tb=short

test-fast:  ## Run tests without coverage for the quickest feedback loop
	pytest tests/ --no-cov -q --tb=short

lint:  ## Lint with ruff (does not modify files)
	ruff check multimind/
	ruff format --check multimind/

format:  ## Auto-format and auto-fix with ruff
	ruff check multimind/ --fix
	ruff format multimind/

typecheck:  ## Run mypy — advisory only (typing migration still in progress)
	-mypy multimind/ --ignore-missing-imports
	@echo "(typecheck is advisory — see [tool.mypy] in pyproject.toml)"

clean:  ## Remove build artifacts and tool caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ \
		coverage.xml test-results*.xml pytest-summary.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:  ## Build sdist + wheel into dist/
	python -m build

publish-test:  ## Upload to TestPyPI (requires TESTPYPI_TOKEN configured in ~/.pypirc)
	twine upload --repository testpypi dist/*

publish:  ## Upload to PyPI (requires PYPI_TOKEN configured in ~/.pypirc)
	twine upload dist/*

docs:  ## Build Sphinx docs into docs/_build/html (requires `make install`)
	sphinx-build -b html docs docs/_build/html
