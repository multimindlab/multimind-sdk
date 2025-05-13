# Quickstart Guide

Get started with MultiMind SDK in minutes. This guide covers the most common use cases and basic functionality.

## Basic Usage

### 1. Create an Agent

```python
import asyncio
from multimind import OpenAIModel, Agent, AgentMemory, CalculatorTool

async def main():
    # Initialize model
    model = OpenAIModel(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create memory and tools
    memory = AgentMemory(max_history=50)
    tools = [CalculatorTool()]
    
    # Create agent
    agent = Agent(
        model=model,
        memory=memory,
        tools=tools,
        system_prompt="You are a helpful AI assistant that can perform calculations."
    )
    
    # Chat with the agent
    response = await agent.chat("What is 123 * 456?")
    print(response)

asyncio.run(main())
```

### 2. Use Prompt Chains

```python
from multimind import OpenAIModel, PromptChain

async def main():
    # Create model
    model = OpenAIModel(model="gpt-3.5-turbo")
    
    # Create prompt chain
    chain = PromptChain(model)
    
    # Add prompts
    chain.add_prompt(
        "analysis",
        "Analyze the following code for security issues: {code}"
    )
    chain.add_prompt(
        "improvements",
        "Based on the analysis, suggest improvements: {analysis}"
    )
    
    # Run chain
    code = """
    def process_user_data(user_input):
        return eval(user_input)
    """
    
    result = await chain.run(code=code)
    print(result)

asyncio.run(main())
```

### 3. Task Runner

```python
from multimind import OpenAIModel, TaskRunner, PromptChain

async def main():
    # Setup
    model = OpenAIModel(model="gpt-3.5-turbo")
    runner = TaskRunner()
    
    # Create research chain
    chain = PromptChain(model)
    chain.add_prompt("research", "Research the topic: {topic}")
    chain.add_prompt("summary", "Summarize the research: {research}")
    
    # Add tasks
    runner.add_task("research", chain, {"topic": "AI in Healthcare"})
    runner.add_task("summary", chain, {"research": "{research}"})
    
    # Run tasks
    results = await runner.run()
    print(results)

asyncio.run(main())
```

### 4. Model Composition Protocol (MCP)

```python
import json
from multimind import OpenAIModel, ClaudeModel, MCPExecutor

async def main():
    # Initialize models
    gpt_model = OpenAIModel(model="gpt-3.5-turbo")
    claude_model = ClaudeModel(model="claude-3-sonnet")
    
    # Create executor
    executor = MCPExecutor()
    executor.register_model("gpt-3.5", gpt_model)
    executor.register_model("claude-3", claude_model)
    
    # Define workflow
    workflow = {
        "version": "1.0.0",
        "models": {
            "gpt-3.5": {"temperature": 0.7},
            "claude-3": {"temperature": 0.7}
        },
        "steps": [
            {
                "name": "analysis",
                "model": "gpt-3.5",
                "prompt": "Analyze: {input}"
            },
            {
                "name": "review",
                "model": "claude-3",
                "prompt": "Review analysis: {analysis}"
            }
        ],
        "connections": [
            {"from": "analysis", "to": "review"}
        ]
    }
    
    # Execute workflow
    result = await executor.execute(workflow, input="AI Ethics")
    print(json.dumps(result, indent=2))

asyncio.run(main())
```

### 5. Usage Tracking

```python
from multimind import (
    OpenAIModel, ClaudeModel,
    UsageTracker, TraceLogger
)

async def main():
    # Initialize trackers
    usage_tracker = UsageTracker()
    trace_logger = TraceLogger()
    
    # Set model costs
    usage_tracker.set_model_cost(
        "gpt-3.5-turbo",
        input_cost_per_token=0.0000015,
        output_cost_per_token=0.000002
    )
    
    # Create models with tracking
    gpt_model = OpenAIModel(
        model="gpt-3.5-turbo",
        usage_tracker=usage_tracker
    )
    claude_model = ClaudeModel(
        model="claude-3-sonnet",
        usage_tracker=usage_tracker
    )
    
    # Start trace
    with trace_logger.start_trace("model_comparison") as trace:
        # Use models
        gpt_response = await gpt_model.generate("Explain AI")
        claude_response = await claude_model.generate("Explain AI")
        
        # Add metadata
        trace.add_metadata({
            "task": "AI explanation",
            "models": ["gpt-3.5-turbo", "claude-3-sonnet"]
        })
    
    # Get usage summary
    summary = usage_tracker.get_summary()
    print(json.dumps(summary, indent=2))

asyncio.run(main())
```

## Next Steps

- Explore [Configuration Options](configuration.md)
- Read the [API Reference](api_reference/README.md)
- Check out the [Examples](../examples/README.md)
- Learn about [Advanced Features](../docs/advanced.md)

## Common Patterns

### Error Handling

```python
from multimind import ModelError

async def safe_generate(model, prompt):
    try:
        response = await model.generate(prompt)
        return response
    except ModelError as e:
        print(f"Model error: {e}")
        # Handle specific error cases
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle other errors
```

### Custom Tools

```python
from multimind import BaseTool

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather for a location"
    
    async def execute(self, location: str) -> str:
        # Implement weather API call
        return f"Weather in {location}: Sunny, 72°F"
```

### Memory Management

```python
from multimind import AgentMemory

# Create memory with custom settings
memory = AgentMemory(
    max_history=100,
    max_tokens=2000,
    include_metadata=True
)

# Add custom metadata
memory.add_metadata({
    "user_id": "123",
    "session_id": "abc"
})
```

## Best Practices

1. **Environment Variables**
   - Always use environment variables for API keys
   - Keep sensitive information out of code

2. **Async/Await**
   - Use async/await for all model operations
   - Handle concurrent requests properly

3. **Error Handling**
   - Implement proper error handling
   - Use specific exception types
   - Log errors appropriately

4. **Resource Management**
   - Monitor token usage
   - Implement rate limiting
   - Clean up resources properly

5. **Testing**
   - Write unit tests for your code
   - Use mock models for testing
   - Test error cases

## Getting Help

- Check the [FAQ](../docs/faq.md)
- Join our [Discord Community](https://discord.gg/multimind)
- Open an issue on [GitHub](https://github.com/multimind-dev/multimind-sdk/issues)
- Contact support at [support@multimind.dev](mailto:support@multimind.dev) 