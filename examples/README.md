# Multimind SDK Examples

This directory contains example scripts demonstrating various features of the Multimind SDK. Each example is designed to showcase specific functionality and can be used as a reference for implementing similar features in your own projects.

## Prerequisites

Before running the examples, make sure you have:

1. Installed the Multimind SDK and its dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

## Available Examples

### 1. Basic Agent (`basic_agent.py`)
Demonstrates how to create and use agents with different models (OpenAI, Claude, Mistral). Shows:
- Agent creation with different models
- Using tools (calculator)
- Memory management
- Running tasks with multiple agents

Run it:
```bash
python basic_agent.py
```

### 2. Prompt Chain (`prompt_chain.py`)
Shows how to use prompt chains for complex reasoning tasks. Features:
- Creating multi-step prompt chains
- Variable substitution
- Conditional execution
- Code review workflow example

Run it:
```bash
python prompt_chain.py
```

### 3. Task Runner (`task_runner.py`)
Demonstrates the task runner for orchestrating complex workflows. Includes:
- Task dependencies
- Retry logic
- Context management
- Research workflow example

Run it:
```bash
python task_runner.py
```

### 4. MCP Workflow (`mcp_workflow.py`)
Shows how to use the Model Composition Protocol for defining and executing complex workflows. Features:
- Workflow definition
- Model registration
- Step types (model, transform, condition)
- Workflow execution

Run it:
```bash
python mcp_workflow.py
```

### 5. Usage Tracking (`usage_tracking.py`)
Demonstrates how to track model usage and costs. Includes:
- Usage monitoring
- Cost tracking
- Trace logging
- Usage reporting

Run it:
```bash
python usage_tracking.py
```

## Example Output

Each example will produce output demonstrating the functionality. For instance:

- `basic_agent.py` will show responses from different models for the same tasks
- `prompt_chain.py` will display a code review with analysis, improvements, and summary
- `task_runner.py` will show a research workflow with topic analysis, research, and recommendations
- `mcp_workflow.py` will demonstrate a multi-model workflow with transformations
- `usage_tracking.py` will show usage statistics and costs

## Customizing Examples

Feel free to modify the examples to suit your needs:

1. Change the models and parameters
2. Modify the prompts and tasks
3. Add your own tools and workflows
4. Adjust the tracking and logging settings

## Best Practices

When using these examples as a reference:

1. Always handle API keys securely
2. Implement proper error handling
3. Use appropriate model parameters
4. Monitor usage and costs
5. Follow the SDK's best practices

## Additional Resources

- [SDK Documentation](../docs/)
- [API Reference](../docs/api.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md) 