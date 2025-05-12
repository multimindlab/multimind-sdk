# Contributing to MultiMind SDK

Thank you for your interest in contributing to MultiMind SDK! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Code Style and Standards](#code-style-and-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Feature Requests and Bug Reports](#feature-requests-and-bug-reports)
10. [Community](#community)
11. [CLI Testing Guidelines](#cli-testing-guidelines)

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
   git remote add upstream https://github.com/original-username/multimind-sdk.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

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
- Maximum line length: 88 characters (Black formatter default)

### Code Formatting

We use the following tools for code formatting and linting:

- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](https://mypy.readthedocs.io/) for type checking

Run the formatters:
```bash
black .
isort .
flake8
mypy .
```

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

- Check the [documentation](https://multimind-sdk.readthedocs.io/)
- Search existing issues
- Join our [Discord community](https://discord.gg/your-invite-link)
- Ask in GitHub Discussions

### Recognition

- Contributors will be listed in the README
- Significant contributions will be highlighted in release notes
- Active contributors may be invited to join the maintainer team

## Additional Resources

- [Project Roadmap](ROADMAP.md)
- [Architecture Overview](docs/architecture.md)
- [Development Guidelines](docs/development.md)
- [Release Process](docs/release_process.md)

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