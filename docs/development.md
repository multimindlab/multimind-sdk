# Development Guide

This guide provides detailed instructions for setting up the development environment and contributing to the MultiMind SDK project.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- CUDA toolkit (for GPU support)
- API keys for supported models (OpenAI, Anthropic, Mistral)

### Initial Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/multimind-dev/multimind-sdk.git
   cd multimind-sdk
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n multimind python=3.8
   conda activate multimind
   ```

3. **Install Development Dependencies**
   ```bash
   # Install in editable mode with development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

5. **Verify Installation**
   ```bash
   # Run tests
   pytest

   # Run examples
   python examples/basic_agent.py
   python examples/prompt_chain.py
   python examples/task_runner.py
   python examples/mcp_workflow.py
   python examples/usage_tracking.py
   ```

## Project Structure

```
multimind-sdk/
├── multimind/                # Main package
│   ├── models/              # Model wrappers
│   ├── router/             # Model routing
│   ├── rag/                # RAG support
│   ├── fine_tuning/        # Training logic
│   ├── agents/             # Agent system
│   ├── orchestration/      # Workflow management
│   ├── mcp/                # Model Composition Protocol
│   ├── integrations/       # Framework integrations
│   ├── logging/            # Monitoring and logging
│   └── cli/                # Command-line interface
├── examples/               # Example scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
└── configs/              # Configuration templates
```

## Development Workflow

### 1. Code Style and Quality

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

#### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Model wrappers
   - Agent system
   - Tools
   - Memory management
   - Configuration

2. **Integration Tests** (`tests/integration/`)
   - Agent workflows
   - MCP execution
   - Prompt chains
   - Task runner
   - Usage tracking

3. **Example Tests** (`tests/examples/`)
   - Basic agent usage
   - Prompt chaining
   - Task running
   - MCP workflows
   - Usage tracking

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/examples/      # Example tests

# Run with coverage
pytest --cov=multimind

# Run specific test file
pytest tests/unit/test_agent.py

# Run specific test
pytest tests/unit/test_agent.py::test_agent_creation
```

#### Writing Tests

Example test structure:

```python
import pytest
from multimind.agents import Agent, AgentMemory
from multimind.models import OpenAIModel

def test_agent_creation():
    # Arrange
    model = OpenAIModel(model="gpt-3.5-turbo")
    memory = AgentMemory(max_history=50)
    
    # Act
    agent = Agent(
        model=model,
        memory=memory,
        system_prompt="You are a helpful assistant."
    )
    
    # Assert
    assert agent.model == model
    assert agent.memory == memory
    assert agent.system_prompt == "You are a helpful assistant."

@pytest.mark.asyncio
async def test_agent_interaction():
    # Arrange
    agent = create_test_agent()  # Helper fixture
    
    # Act
    response = await agent.run("What is 2+2?")
    
    # Assert
    assert response is not None
    assert "4" in response.lower()
```

### 3. Examples

The `examples/` directory contains working examples of all major features:

1. **Basic Agent** (`basic_agent.py`)
   - Agent creation
   - Model usage
   - Tool integration
   - Memory management

2. **Prompt Chain** (`prompt_chain.py`)
   - Multi-step reasoning
   - Variable substitution
   - Code review workflow

3. **Task Runner** (`task_runner.py`)
   - Task dependencies
   - Research workflow
   - Context management

4. **MCP Workflow** (`mcp_workflow.py`)
   - Workflow definition
   - Model composition
   - Step execution

5. **Usage Tracking** (`usage_tracking.py`)
   - Usage monitoring
   - Cost tracking
   - Export/reporting

To run examples:
```bash
# Run all examples
python examples/basic_agent.py
python examples/prompt_chain.py
python examples/task_runner.py
python examples/mcp_workflow.py
python examples/usage_tracking.py

# Run with specific model
OPENAI_API_KEY=your_key python examples/basic_agent.py
```

### 4. Documentation

#### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public APIs
- Add examples in docstrings

Example:
```python
def create_agent(
    model: BaseModel,
    memory: Optional[AgentMemory] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: str = "You are a helpful assistant."
) -> Agent:
    """Create a new agent with the specified configuration.

    Args:
        model: The language model to use.
        memory: Optional memory for conversation history.
        tools: Optional list of tools for the agent.
        system_prompt: The system prompt to use.

    Returns:
        An initialized Agent instance.

    Example:
        >>> model = OpenAIModel(model="gpt-3.5-turbo")
        >>> agent = create_agent(model, system_prompt="You are a math tutor.")
        >>> response = await agent.run("What is 2+2?")
        >>> print(response)
        "The answer is 4."
    """
    pass
```

#### User Documentation

- Update README.md for significant changes
- Add/update docstrings
- Include usage examples
- Document configuration options

### 5. Version Control

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

### 6. Pull Requests

1. Create feature/fix branch
2. Make changes
3. Run tests and checks:
   ```bash
   # Run all checks
   pytest
   black . --check
   isort . --check
   flake8
   mypy .
   ```
4. Update documentation
5. Create PR with template
6. Request review

## Advanced Development

### Adding New Features

1. **New Model Wrapper**
   ```python
   from multimind.models.base import BaseModel

   class NewModelWrapper(BaseModel):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           
       async def generate(self, prompt: str) -> str:
           # Implementation
           pass
   ```

2. **New Agent Tool**
   ```python
   from multimind.agents.tools.base import BaseTool

   class NewTool(BaseTool):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           
       async def execute(self, input_data: Any) -> Any:
           # Implementation
           pass
   ```

3. **New MCP Step Type**
   ```python
   from multimind.mcp.base import BaseStep

   class NewStepType(BaseStep):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           
       async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
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

1. **API Key Issues**
   - Check environment variables
   - Verify API key permissions
   - Check rate limits

2. **Model Loading Issues**
   - Check model availability
   - Verify model configuration
   - Check GPU memory (if using)

3. **Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

### Getting Help

- Check [documentation](https://multimind-sdk.readthedocs.io/)
- Search [issues](https://github.com/multimind-dev/multimind-sdk/issues)
- Join [Discord](https://discord.gg/your-invite-link)
- Ask in [Discussions](https://github.com/multimind-dev/multimind-sdk/discussions)

## Contributor License Agreement (CLA)

Before contributing to the MultiMind SDK, you must sign our Contributor License Agreement. This agreement ensures that the project has the necessary rights to use, modify, and distribute your contributions.

### Why We Need a CLA

The CLA serves several important purposes:
1. Establishes clear terms for contributions
2. Protects both contributors and the project
3. Ensures the project can be used under its chosen license
4. Provides a record of contributor consent

### CLA Process

1. **Review the Agreement**
   - Read the [CLA document](../CLA.md)
   - Understand your rights and obligations
   - Review the license terms

2. **Sign the Agreement**
   - Individual contributors: Sign through [CLA Assistant](https://cla-assistant.io/multimind-dev/multimind-sdk)
   - Corporate contributors: Contact the project maintainers for a corporate CLA

3. **Link Your GitHub Account**
   - Connect your GitHub account to CLA Assistant
   - This allows automatic verification of your contributions

4. **Verify Status**
   - Check your CLA status in pull requests
   - CLA Assistant will comment on PRs with status
   - Ensure your email matches in Git config and CLA

### CLA Requirements

1. **For Individual Contributors**
   - Must be 18 years or older
   - Must have authority to grant the rights
   - Must provide valid contact information
   - Must use consistent identity across contributions

2. **For Corporate Contributors**
   - Must have authority to bind the organization
   - Must provide company details
   - Must designate authorized contributors
   - Must maintain current contact information

3. **For All Contributors**
   - Must agree to the terms of the MIT License
   - Must warrant contributions are original work
   - Must not include third-party code without permission
   - Must disclose any relevant patents or IP

### CLA Enforcement

- All pull requests require CLA verification
- CLA Assistant automatically checks status
- Maintainers will not merge PRs without CLA
- Existing contributors must sign for new contributions

### Updating Your CLA

If you need to update your CLA information:
1. Contact the project maintainers
2. Provide updated information
3. Sign a new agreement if necessary
4. Update your Git configuration

### CLA FAQ

1. **Do I need to sign for each contribution?**
   - No, one signature covers all contributions
   - Keep your CLA information current

2. **What if I change employers?**
   - Update your CLA if contributing on behalf of a company
   - Personal contributions remain covered

3. **Can I revoke my CLA?**
   - No, but you can stop contributing
   - Existing contributions remain licensed

4. **What about small changes?**
   - All contributions require CLA
   - No exceptions for size or type

For questions about the CLA process, contact the project maintainers at [legal@multimind.dev](mailto:legal@multimind.dev).

For more details, see the [Architecture Overview](architecture.md) and [API Reference](api.md). 