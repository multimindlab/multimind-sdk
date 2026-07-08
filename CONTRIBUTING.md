# Contributing to MultiMind SDK

Thank you for your interest in contributing to MultiMind SDK! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Good First Areas](#good-first-areas)
5. [Contribution Workflow](#contribution-workflow)
6. [Code Style and Standards](#code-style-and-standards)
7. [Testing](#testing)
8. [Documentation](#documentation)
9. [Pull Request Process](#pull-request-process)
10. [Feature Requests and Bug Reports](#feature-requests-and-bug-reports)
11. [Community](#community)
12. [CLI Testing Guidelines](#cli-testing-guidelines)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/multimind-sdk.git
   cd multimind-sdk
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/multimindlab/multimind-sdk.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies + pre-commit hooks in one shot:
   ```bash
   make install
   ```

   If you don't have GNU `make`, the equivalent two commands are:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. (Optional) Install every extras group for end-to-end work on RAG, agents,
   compliance, fine-tuning, etc.:
   ```bash
   make install-all
   ```

### Common dev tasks (via `make`)

| Command           | What it does                                                          |
| ----------------- | --------------------------------------------------------------------- |
| `make help`       | Show every target with a one-line description.                        |
| `make test`       | Run tests excluding `integration`, `requires_api_key`, `slow` markers. |
| `make test-all`   | Run the full test suite.                                              |
| `make lint`       | `ruff check` + `ruff format --check` (no file mutation).              |
| `make format`     | `ruff check --fix` + `ruff format` (mutates files).                   |
| `make typecheck`  | Run mypy (advisory — typing migration is in progress).                |
| `make clean`      | Remove build artifacts and tool caches.                               |
| `make build`      | Build sdist + wheel into `dist/`.                                     |

### Pre-commit

`pre-commit install` wires the hooks in `.pre-commit-config.yaml` into
`git commit`. They run automatically on every commit and currently cover:

- Whitespace / EOF / merge-conflict / private-key / large-file checks
- `ruff check --fix` (lints + safe auto-fixes)
- `ruff format` (formatter — replaces black; settings in `[tool.ruff.format]`)

Mypy is *not* in pre-commit yet — see the comment in
`.pre-commit-config.yaml` for the rationale. Use `make typecheck` to run it
manually.

If a hook modifies files (e.g. `ruff --fix` cleans an import), the commit
fails with a clear message. Re-run `git add` for the modified files and
commit again.

## Good First Areas

Approachable places to make a genuinely useful first contribution:

- **Examples** (`examples/`) — small, runnable scripts demonstrating one feature.
  Several offline examples ([examples/compliance/guarded_model.py](examples/compliance/guarded_model.py),
  [examples/observability/cost_tracking.py](examples/observability/cost_tracking.py),
  [examples/evaluation/hallucination_detection.py](examples/evaluation/hallucination_detection.py))
  show the pattern; a new example needs no deep knowledge of the internals.
- **Documentation** (`docs/`, README, this file) — fixing inaccuracies, adding
  recipes to [docs/cookbook.md](docs/cookbook.md), or improving the
  [non-technical guide](docs/getting-started-non-technical.md). If a doc
  confused you, that confusion is the bug report.
- **Vector-store backends** (`multimind/vector_store/`) — many of the 40+
  backends are client-backed stubs whose methods still `raise NotImplementedError`
  (e.g. `cassandra.py`, `azuresearch.py`, `clickhouse.py`, `tiledb.py`).
  Picking one and implementing its methods against the real client library is
  self-contained: the interface to satisfy is in
  `multimind/vector_store/base.py`, and working references include the FAISS
  and Chroma backends.

Also check issues labeled `good first issue` on
[GitHub](https://github.com/multimindlab/multimind-sdk/issues).

## Contribution Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-fix-name
   ```

2. Make your changes following our [code style guidelines](#code-style-and-standards)

3. Run tests and ensure they pass:
   ```bash
   pytest
   ```

4. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request from your fork to the main repository

## Code Style and Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Document all public functions, classes, and methods using docstrings
- Maximum line length: 100 characters (configured in `[tool.ruff]`)

### Code Formatting

We standardize on a single tool — **Ruff** — for linting, import sorting,
and formatting. There is no separate `black` / `isort` / `flake8` step.

- `ruff check` — lint + import sorting (replaces flake8 + isort)
- `ruff format` — code formatter (replaces black)
- `mypy` — type checking (advisory; opt-in via `make typecheck`)

Run them via the Makefile (recommended):

```bash
make format     # auto-fix + reformat (mutates files)
make lint       # check only (no file changes)
make typecheck  # advisory mypy run
```

Or invoke ruff directly:

```bash
ruff check multimind/        # report lint issues
ruff check multimind/ --fix  # apply safe auto-fixes
ruff format multimind/       # apply formatter
```

All settings live in `pyproject.toml` under `[tool.ruff]` and
`[tool.ruff.format]` — single source of truth for CI, pre-commit, and
local runs.

### Documentation Style

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include examples in docstrings where appropriate
- Keep documentation up-to-date with code changes

## Testing

### Writing Tests

- Write tests for all new features and bug fixes
- Use [pytest](https://docs.pytest.org/) for testing
- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Include both unit tests and integration tests

Example test structure:
```python
def test_feature_name():
    # Arrange
    setup_code()
    
    # Act
    result = function_to_test()
    
    # Assert
    assert result == expected_value
```

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_file.py
```

Run with coverage:
```bash
pytest --cov=multimind
```

## Documentation

### Code Documentation

- Document all public APIs
- Include type hints
- Provide clear examples
- Update docstrings when changing code

### User Documentation

- Update README.md for significant changes
- Add/update docstrings for new features
- Include usage examples
- Document configuration options

## Pull Request Process

1. Ensure your code follows our style guidelines
2. Update documentation for any new features
3. Add tests for new functionality
4. Ensure all tests pass
5. Update the changelog
6. Fill out the PR template
7. Request review from maintainers

### PR Template

```markdown
## Description
[Describe your changes here]

## Related Issues
[Link to related issues]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Changelog updated
```

## Feature Requests and Bug Reports

### Feature Requests

1. Check existing issues to avoid duplicates
2. Use the feature request template
3. Provide clear description and use cases
4. Include potential implementation ideas

### Bug Reports

1. Use the bug report template
2. Include steps to reproduce
3. Provide expected and actual behavior
4. Include environment details
5. Add error messages and logs

## Community

### Communication Channels

- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For general discussions
- Discord: For real-time communication
- Email: For private matters

### Getting Help

- Check the [documentation](docs/README.md)
- Search existing issues
- Join our [Discord community](https://discord.gg/K64U65je7h)
- Ask in GitHub Discussions

### Recognition

- Contributors will be listed in the README
- Significant contributions will be highlighted in release notes
- Active contributors may be invited to join the maintainer team

## Additional Resources

- [Project Roadmap](ROADMAP.md)
- [Architecture Overview](docs/architecture.md)
- [Development Guidelines](docs/development.md)
- [Docs Index](docs/README.md)

## CLI Testing Guidelines

All new CLI features or changes **must** include corresponding tests in `tests/test_cli.py`.

- Use `pytest` and `pytest-mock` for mocking SDK and external dependencies.
- Use `click.testing.CliRunner` to invoke CLI commands and check outputs.
- Test both success and failure cases, including edge cases and user prompts.
- For destructive actions (like `delete`), test both confirmation and abort scenarios.

### Minimal Example

```python
from click.testing import CliRunner
from multimind.cli import cli

def test_train_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--help'])
    assert result.exit_code == 0
    assert "Fine-tune a model" in result.output

def test_train_with_mock(mocker):
    runner = CliRunner()
    mock_tuner = mocker.patch('multimind.cli.UniPELTTuner')
    instance = mock_tuner.return_value
    instance.train.return_value = None
    result = runner.invoke(cli, ['train', '--config', 'dummy.yaml'])
    assert result.exit_code == 0 or result.exit_code == 1
    instance.train.assert_called()
```

See `tests/test_cli.py` for more comprehensive examples.

Thank you for contributing to MultiMind SDK! Your help makes the project better for everyone. 