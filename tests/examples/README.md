# MultiMind SDK Examples Testing

This directory contains comprehensive tests for all examples in the `examples/` directory of the MultiMind SDK.

## Test Structure

The test files follow the same directory structure as the examples:

```
tests/examples/
├── cli/                    # Tests for CLI examples
│   ├── test_basic_agent.py
│   ├── test_chat_with_gpt.py
│   └── ...
├── api/                    # Tests for API examples
│   ├── test_ensemble_api.py
│   ├── test_rag_example.py
│   └── ...
├── compliance/             # Tests for compliance examples
│   ├── test_healthcare_compliance.py
│   └── ...
├── model_management/       # Tests for model management examples
├── memory/                 # Tests for memory examples
├── rag/                    # Tests for RAG examples
├── vector_store/           # Tests for vector store examples
├── mcp/                    # Tests for MCP examples
├── multi_modal/            # Tests for multi-modal examples
├── fine_tuning/            # Tests for fine-tuning examples
├── client/                 # Tests for client examples
├── config/                 # Tests for config examples
├── agents/                 # Tests for agents examples
├── pipeline/               # Tests for pipeline examples
├── context_transfer/       # Tests for context transfer examples
├── non_transformer/        # Tests for non-transformer examples
├── advanced/               # Tests for advanced examples
├── streamlit-ui/           # Tests for Streamlit UI examples
├── moe/                    # Tests for MoE examples
├── data/                   # Tests for data examples
├── model_conversion/       # Tests for model conversion examples
├── observability/          # Tests for observability examples
├── ensemble/               # Tests for ensemble examples
├── hybrid_workflow/        # Tests for hybrid workflow examples
└── test_all_examples.py    # Comprehensive test runner
```

## Test Approach

### 1. Mock-Based Testing
All tests use comprehensive mocking to avoid external dependencies:
- **API Services**: OpenAI, Anthropic, HuggingFace, etc.
- **Vector Databases**: Chroma, Pinecone, Weaviate, etc.
- **ML Libraries**: PyTorch, Transformers, etc.
- **External Services**: HTTP requests, file systems, etc.

### 2. Test Categories

#### Import Tests
- Verify that example modules can be imported without errors
- Check for missing dependencies
- Validate module structure

#### Functionality Tests
- Test main functions and key classes
- Verify async/sync function execution
- Test error handling and edge cases

#### Integration Tests
- Test interactions between components
- Verify data flow and transformations
- Test configuration and setup

#### Structure Tests
- Verify example file structure
- Check for required components
- Validate documentation and comments

### 3. Test Patterns

#### Mock Classes
```python
class MockOpenAIModel:
    """Mock OpenAI model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock OpenAI response to: {prompt}"
    
    async def chat(self, messages, **kwargs):
        return "Mock OpenAI chat response"
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)
```

#### Test Fixtures
```python
@pytest.fixture
def mock_models():
    """Fixture to provide mock models."""
    return {
        "gpt-4": MockOpenAIModel("gpt-4"),
        "claude-3": MockClaudeModel("claude-3"),
        "mistral": MockMistralModel("mistral")
    }
```

#### Async Test Functions
```python
@pytest.mark.asyncio
async def test_example_functionality():
    """Test example functionality."""
    with patch('examples.module.RealClass', MockClass):
        result = await example_function()
        assert result is not None
```

## Running Tests

### Individual Test Files
```bash
# Test a specific example category
python -m pytest tests/examples/cli/test_basic_agent.py -v

# Test all CLI examples
python -m pytest tests/examples/cli/ -v

# Test all examples
python -m pytest tests/examples/ -v
```

### Comprehensive Test Runner
```bash
# Run the comprehensive test runner
python tests/examples/test_all_examples.py
```

### Test with Coverage
```bash
# Run tests with coverage reporting
python -m pytest tests/examples/ --cov=examples --cov-report=html
```

## Test Results

### Success Criteria
- **Import Tests**: All modules import without errors
- **Functionality Tests**: Core functions execute successfully
- **Integration Tests**: Components work together correctly
- **Structure Tests**: Examples follow expected patterns

### Common Issues and Fixes

#### 1. Missing Dependencies
**Issue**: `ModuleNotFoundError` for optional dependencies
**Fix**: Use conditional imports and mock missing modules

#### 2. API Key Requirements
**Issue**: Examples require API keys for external services
**Fix**: Mock API clients and responses

#### 3. File System Dependencies
**Issue**: Examples require specific files or directories
**Fix**: Create temporary test files or mock file operations

#### 4. Async/Sync Mismatches
**Issue**: Mixing async and sync code
**Fix**: Ensure consistent async/await patterns

#### 5. Environment Variables
**Issue**: Examples require specific environment variables
**Fix**: Mock environment or set test variables

## Adding New Tests

### 1. Create Test File
Create a test file in the appropriate directory:
```python
"""
Tests for example_name.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.category.example_name import main
```

### 2. Create Mock Classes
Define mock classes for external dependencies:
```python
class MockExampleModel:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock response to: {prompt}"
```

### 3. Write Test Functions
Create comprehensive test functions:
```python
@pytest.mark.asyncio
async def test_example_imports():
    """Test that the example can be imported."""
    try:
        from examples.category import example_name
        assert example_name is not None
    except ImportError as e:
        pytest.skip(f"Example not available: {e}")

@pytest.mark.asyncio
async def test_example_main():
    """Test that the main function can be called."""
    with patch('examples.category.example_name.RealClass', MockClass):
        try:
            await main()
            assert True
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")
```

### 4. Add to Test Runner
Update `test_all_examples.py` to include the new example.

## Test Maintenance

### Regular Updates
- Update mocks when external APIs change
- Add tests for new example features
- Remove tests for deprecated examples

### Performance Monitoring
- Track test execution times
- Monitor for flaky tests
- Optimize slow test cases

### Documentation
- Keep test documentation up to date
- Document new test patterns
- Update troubleshooting guides

## Troubleshooting

### Common Test Failures

#### Import Errors
```bash
# Check if module exists
python -c "import examples.category.example_name"

# Check dependencies
pip list | grep dependency_name
```

#### Mock Issues
```python
# Ensure mocks are applied correctly
with patch('module.path.ClassName', MockClass):
    # Test code here
```

#### Async Issues
```python
# Use proper async test decorator
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Debug Mode
```bash
# Run tests with debug output
python -m pytest tests/examples/ -v -s --tb=long
```

## Contributing

When adding new examples or modifying existing ones:

1. **Create Tests**: Add comprehensive tests for new functionality
2. **Update Mocks**: Ensure mocks cover all external dependencies
3. **Document Changes**: Update this README with new test patterns
4. **Run Full Suite**: Ensure all tests pass before submitting

## Test Coverage Goals

- **Import Coverage**: 100% of examples importable
- **Functionality Coverage**: 90% of main functions testable
- **Integration Coverage**: 80% of component interactions tested
- **Error Handling**: 100% of expected error cases covered

## Future Improvements

1. **Automated Test Generation**: Generate tests from example metadata
2. **Performance Benchmarking**: Track example execution performance
3. **Dependency Analysis**: Automatically detect and mock dependencies
4. **Visual Test Reports**: Generate HTML reports with detailed results
5. **CI/CD Integration**: Automated testing in continuous integration 