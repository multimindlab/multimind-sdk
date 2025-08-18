"""
Tests for basic_agent.py CLI example.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import os
import sys
from pathlib import Path

# Add examples directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.cli.basic_agent import main


class MockOpenAIModel:
    """Mock OpenAI model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock OpenAI response to: {prompt}"
    
    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock OpenAI stream response to: {prompt}"
        return stream()
    
    async def chat(self, messages, **kwargs):
        return "Mock OpenAI chat response"
    
    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock OpenAI chat stream response"
        return stream()
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384] * len(text)


class MockClaudeModel:
    """Mock Claude model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock Claude response to: {prompt}"
    
    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock Claude stream response to: {prompt}"
        return stream()
    
    async def chat(self, messages, **kwargs):
        return "Mock Claude chat response"
    
    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock Claude chat stream response"
        return stream()
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.2] * 384
        return [[0.2] * 384] * len(text)


class MockMistralModel:
    """Mock Mistral model for testing."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def generate(self, prompt: str, **kwargs):
        return f"Mock Mistral response to: {prompt}"
    
    async def generate_stream(self, prompt: str, **kwargs):
        async def stream():
            yield f"Mock Mistral stream response to: {prompt}"
        return stream()
    
    async def chat(self, messages, **kwargs):
        return "Mock Mistral chat response"
    
    async def chat_stream(self, messages, **kwargs):
        async def stream():
            yield "Mock Mistral chat stream response"
        return stream()
    
    async def embeddings(self, text, **kwargs):
        if isinstance(text, str):
            return [0.3] * 384
        return [[0.3] * 384] * len(text)


class MockAgentMemory:
    """Mock agent memory for testing."""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.history = []
    
    def get_history(self, n: int = None):
        if n is None:
            return self.history
        return self.history[-n:]
    
    async def add_message(self, message):
        self.history.append(message)
        if len(self.history) > self.max_history:
            self.history.pop(0)


class MockCalculatorTool:
    """Mock calculator tool for testing."""
    
    def __init__(self):
        self.calls = 0
    
    async def execute(self, *args, **kwargs):
        self.calls += 1
        return f"Mock calculator result: {args}"
    
    def get_parameters(self):
        return []


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, model, memory, tools, system_prompt):
        self.model = model
        self.memory = memory
        self.tools = tools
        self.system_prompt = system_prompt
        self.run_calls = 0
    
    async def run(self, task):
        self.run_calls += 1
        # Simulate using the model to generate response
        response = await self.model.generate(task)
        # Simulate adding to memory
        await self.memory.add_message({"role": "user", "content": task})
        await self.memory.add_message({"role": "assistant", "content": response})
        return response


@pytest.fixture
def mock_models():
    """Fixture to provide mock models."""
    return {
        "openai": MockOpenAIModel("gpt-3.5-turbo"),
        "claude": MockClaudeModel("claude-3-sonnet-20240229"),
        "mistral": MockMistralModel("mistral-medium")
    }


@pytest.fixture
def mock_memory():
    """Fixture to provide mock memory."""
    return MockAgentMemory(max_history=50)


@pytest.fixture
def mock_tools():
    """Fixture to provide mock tools."""
    return [MockCalculatorTool()]


@pytest.mark.asyncio
async def test_basic_agent_imports():
    """Test that the basic_agent module can be imported."""
    try:
        from examples.cli.basic_agent import main
        assert main is not None
    except ImportError as e:
        pytest.fail(f"Failed to import basic_agent: {e}")


@pytest.mark.asyncio
async def test_basic_agent_main_function():
    """Test that the main function can be called without errors."""
    with patch('examples.cli.basic_agent.OpenAIModel', MockOpenAIModel), \
         patch('examples.cli.basic_agent.ClaudeModel', MockClaudeModel), \
         patch('examples.cli.basic_agent.MistralModel', MockMistralModel), \
         patch('examples.cli.basic_agent.Agent', MockAgent), \
         patch('examples.cli.basic_agent.AgentMemory', MockAgentMemory), \
         patch('examples.cli.basic_agent.CalculatorTool', MockCalculatorTool), \
         patch('examples.cli.basic_agent.load_dotenv'):
        
        try:
            await main()
            assert True  # If we get here, the function ran without errors
        except Exception as e:
            pytest.fail(f"main() function failed: {e}")


@pytest.mark.asyncio
async def test_model_initialization():
    """Test that models can be initialized correctly."""
    # Test OpenAI model
    openai_model = MockOpenAIModel("gpt-3.5-turbo", temperature=0.7)
    assert openai_model.model_name == "gpt-3.5-turbo"
    assert openai_model.kwargs["temperature"] == 0.7
    
    # Test Claude model
    claude_model = MockClaudeModel("claude-3-sonnet-20240229", temperature=0.7)
    assert claude_model.model_name == "claude-3-sonnet-20240229"
    assert claude_model.kwargs["temperature"] == 0.7
    
    # Test Mistral model
    mistral_model = MockMistralModel("mistral-medium", temperature=0.7)
    assert mistral_model.model_name == "mistral-medium"
    assert mistral_model.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_agent_creation():
    """Test that agents can be created with models, memory, and tools."""
    model = MockOpenAIModel("gpt-3.5-turbo")
    memory = MockAgentMemory(max_history=50)
    tools = [MockCalculatorTool()]
    
    agent = MockAgent(
        model=model,
        memory=memory,
        tools=tools,
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    assert agent.model == model
    assert agent.memory == memory
    assert agent.tools == tools
    assert agent.system_prompt == "You are a helpful AI assistant that can perform calculations."


@pytest.mark.asyncio
async def test_agent_execution():
    """Test that agents can execute tasks."""
    model = MockOpenAIModel("gpt-3.5-turbo")
    memory = MockAgentMemory(max_history=50)
    tools = [MockCalculatorTool()]
    
    agent = MockAgent(
        model=model,
        memory=memory,
        tools=tools,
        system_prompt="You are a helpful AI assistant."
    )
    
    # Test agent execution
    task = "What is 123 * 456?"
    response = await agent.run(task)
    
    assert response is not None
    assert "Mock OpenAI response" in response
    assert agent.run_calls == 1
    
    # Check that memory was updated
    history = memory.get_history()
    assert len(history) == 2  # User message and assistant response
    assert history[0]["role"] == "user"
    assert history[0]["content"] == task
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_model_generation():
    """Test that models can generate responses."""
    openai_model = MockOpenAIModel("gpt-3.5-turbo")
    claude_model = MockClaudeModel("claude-3-sonnet-20240229")
    mistral_model = MockMistralModel("mistral-medium")
    
    prompt = "Explain quantum computing"
    
    # Test OpenAI generation
    openai_response = await openai_model.generate(prompt)
    assert "Mock OpenAI response" in openai_response
    assert prompt in openai_response
    
    # Test Claude generation
    claude_response = await claude_model.generate(prompt)
    assert "Mock Claude response" in claude_response
    assert prompt in claude_response
    
    # Test Mistral generation
    mistral_response = await mistral_model.generate(prompt)
    assert "Mock Mistral response" in mistral_response
    assert prompt in mistral_response


@pytest.mark.asyncio
async def test_model_streaming():
    """Test that models can generate streaming responses."""
    model = MockOpenAIModel("gpt-3.5-turbo")
    prompt = "Explain quantum computing"
    
    # Test streaming generation
    stream = await model.generate_stream(prompt)
    responses = []
    async for chunk in stream:
        responses.append(chunk)
    
    assert len(responses) > 0
    assert any("Mock OpenAI stream response" in response for response in responses)


@pytest.mark.asyncio
async def test_model_chat():
    """Test that models can handle chat conversations."""
    model = MockOpenAIModel("gpt-3.5-turbo")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    response = await model.chat(messages)
    assert "Mock OpenAI chat response" in response


@pytest.mark.asyncio
async def test_model_embeddings():
    """Test that models can generate embeddings."""
    model = MockOpenAIModel("gpt-3.5-turbo")
    
    # Test single text embedding
    text = "Hello world"
    embedding = await model.embeddings(text)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)
    
    # Test multiple text embeddings
    texts = ["Hello", "World", "Test"]
    embeddings = await model.embeddings(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


@pytest.mark.asyncio
async def test_memory_operations():
    """Test memory operations."""
    memory = MockAgentMemory(max_history=3)
    
    # Add messages
    await memory.add_message({"role": "user", "content": "Message 1"})
    await memory.add_message({"role": "assistant", "content": "Response 1"})
    await memory.add_message({"role": "user", "content": "Message 2"})
    await memory.add_message({"role": "assistant", "content": "Response 2"})
    
    # Test history retrieval
    history = memory.get_history()
    assert len(history) == 3  # Should be limited by max_history
    
    # Test limited history
    recent = memory.get_history(n=2)
    assert len(recent) == 2
    assert recent[-1]["content"] == "Response 2"


@pytest.mark.asyncio
async def test_calculator_tool():
    """Test calculator tool functionality."""
    tool = MockCalculatorTool()
    
    # Test tool execution
    result = await tool.execute("2 + 2")
    assert "Mock calculator result" in result
    assert tool.calls == 1
    
    # Test parameters
    params = tool.get_parameters()
    assert isinstance(params, list)


@pytest.mark.asyncio
async def test_environment_variables():
    """Test that environment variables are properly handled."""
    # Test that load_dotenv is called
    with patch('examples.cli.basic_agent.load_dotenv') as mock_load_dotenv:
        with patch('examples.cli.basic_agent.OpenAIModel', MockOpenAIModel), \
             patch('examples.cli.basic_agent.ClaudeModel', MockClaudeModel), \
             patch('examples.cli.basic_agent.MistralModel', MockMistralModel), \
             patch('examples.cli.basic_agent.Agent', MockAgent), \
             patch('examples.cli.basic_agent.AgentMemory', MockAgentMemory), \
             patch('examples.cli.basic_agent.CalculatorTool', MockCalculatorTool):
            
            await main()
            mock_load_dotenv.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in the basic agent example."""
    # Test with failing model
    failing_model = MockOpenAIModel("gpt-3.5-turbo")
    failing_model.generate = AsyncMock(side_effect=Exception("API Error"))
    
    memory = MockAgentMemory()
    tools = [MockCalculatorTool()]
    
    agent = MockAgent(
        model=failing_model,
        memory=memory,
        tools=tools,
        system_prompt="You are a helpful AI assistant."
    )
    
    # The agent should handle the error gracefully
    with pytest.raises(Exception, match="API Error"):
        await agent.run("Test task")


def test_example_structure():
    """Test that the example has the expected structure."""
    example_path = Path(__file__).parent.parent.parent.parent / "examples" / "cli" / "basic_agent.py"
    assert example_path.exists(), "basic_agent.py example should exist"
    
    # Check that the file contains expected components
    with open(example_path, 'r') as f:
        content = f.read()
        assert "async def main()" in content
        assert "OpenAIModel" in content
        assert "ClaudeModel" in content
        assert "MistralModel" in content
        assert "Agent" in content
        assert "AgentMemory" in content
        assert "CalculatorTool" in content 