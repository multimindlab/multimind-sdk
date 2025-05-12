# Development Guide

This guide provides detailed instructions for setting up the development environment and contributing to the MultiMind SDK project.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- CUDA toolkit (for GPU support)

### Initial Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/multimind-dev/multimind-sdk.git
   cd multimind-sdk
   ```

2. **Add the upstream repository**
   ```bash
   git remote add upstream https://github.com/multimind-dev/multimind-sdk.git
   ```

3. **Create Virtual Environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n multimind python=3.8
   conda activate multimind
   ```

4. **Install Development Dependencies**
   ```bash
   # Install in editable mode with development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   # Run tests
   pytest

   # Check code style
   black . --check
   isort . --check
   flake8
   mypy .
   ```

## Project Structure

```
multimind-sdk/
├── docs/                    # Documentation
├── examples/               # Example notebooks and scripts
├── multimind/             # Main package
│   ├── fine_tuning/      # Fine-tuning implementations
│   ├── integrations/     # Framework integrations
│   ├── utils/           # Utility functions
│   └── __init__.py
├── tests/                # Test suite
├── .pre-commit-config.yaml
├── pyproject.toml
├── setup.py
└── README.md
```

## Development Workflow

### 1. Code Style

We use several tools to maintain code quality:

```bash
# Format code
black .
isort .

# Check code style
flake8
mypy .

# Run pre-commit hooks
pre-commit run --all-files
```

### 2. Testing

#### Writing Tests

- Place tests in the `tests/` directory
- Follow naming convention: `test_*.py`
- Use pytest fixtures for common setup
- Include both unit and integration tests

Example test:
```python
import pytest
from multimind.fine_tuning import UniPELTPlusTuner

def test_unipelt_tuner_initialization():
    # Arrange
    model_name = "bert-base-uncased"
    output_dir = "./test_output"
    
    # Act
    tuner = UniPELTPlusTuner(
        base_model_name=model_name,
        output_dir=output_dir
    )
    
    # Assert
    assert tuner.model_name == model_name
    assert tuner.output_dir == output_dir
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run with coverage
pytest --cov=multimind

# Run specific test
pytest tests/test_file.py::test_name
```

### 3. Documentation

#### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public APIs
- Add examples in docstrings

Example:
```python
def train_model(
    model: torch.nn.Module,
    train_data: Dataset,
    epochs: int = 10
) -> Dict[str, float]:
    """Train a model on the given dataset.

    Args:
        model: The model to train.
        train_data: Training dataset.
        epochs: Number of training epochs.

    Returns:
        Dictionary containing training metrics.

    Example:
        >>> model = MyModel()
        >>> metrics = train_model(model, train_dataset, epochs=5)
        >>> print(metrics['loss'])
        0.123
    """
    pass
```

#### User Documentation

- Update README.md for significant changes
- Add/update docstrings
- Include usage examples
- Document configuration options

### 4. Version Control

#### Branch Strategy

- `main`: Production-ready code
- `develop`: Development branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

#### Commit Messages

Follow conventional commits:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Testing
- `chore`: Maintenance

### 5. Pull Requests

1. Create feature/fix branch
2. Make changes
3. Run tests and checks
4. Update documentation
5. Create PR with template
6. Request review

## Advanced Development

### Adding New Features

1. **New PEFT Method**
   ```python
   from multimind.fine_tuning.base import BasePEFTMethod

   class MyPEFTMethod(BasePEFTMethod):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           
       def apply(self, model: torch.nn.Module) -> torch.nn.Module:
           # Implementation
           pass
   ```

2. **New Framework Adapter**
   ```python
   from multimind.integrations.base import BaseModelAdapter

   class MyFrameworkAdapter(BaseModelAdapter):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           
       def generate(self, prompt: str) -> str:
           # Implementation
           pass
   ```

### Performance Optimization

1. **Profiling**
   ```bash
   # Using cProfile
   python -m cProfile -o output.prof script.py
   
   # Using line_profiler
   kernprof -l script.py
   ```

2. **Memory Profiling**
   ```bash
   # Using memory_profiler
   mprof run script.py
   mprof plot
   ```

### Debugging

1. **Using pdb**
   ```python
   import pdb; pdb.set_trace()
   ```

2. **Using logging**
   ```python
   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   
   logger.debug("Debug message")
   logger.info("Info message")
   logger.warning("Warning message")
   logger.error("Error message")
   ```

## CI/CD Pipeline

### GitHub Actions

Workflow files in `.github/workflows/`:
- `test.yml`: Run tests
- `lint.yml`: Check code style
- `docs.yml`: Build documentation
- `release.yml`: Create releases

### Local CI

```bash
# Run all checks
./scripts/check.sh

# Run specific checks
./scripts/check.sh test
./scripts/check.sh lint
./scripts/check.sh docs
```

## Release Process

1. Update version in `setup.py`
2. Update changelog
3. Create release branch
4. Run full test suite
5. Build documentation
6. Create GitHub release
7. Publish to PyPI

## Troubleshooting

### Common Issues

1. **CUDA Issues**
   - Check CUDA version compatibility
   - Verify GPU availability
   - Check PyTorch CUDA installation

2. **Dependency Conflicts**
   - Use virtual environment
   - Check dependency versions
   - Update requirements.txt

3. **Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

### Getting Help

- Check [documentation](https://multimind-sdk.readthedocs.io/)
- Search [issues](https://github.com/multimind-dev/multimind-sdk/issues)
- Join [Discord](https://discord.gg/your-invite-link)
- Ask in [Discussions](https://github.com/multimind-dev/multimind-sdk/discussions) 