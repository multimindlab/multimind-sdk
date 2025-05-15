# Basic Usage Guide

This guide walks you through the most common tasks with MultiMind SDK.

## 1. Create an Agent

```python
from multimind.models import OpenAIModel
from multimind.agents import Agent, AgentMemory
from multimind.agents.tools import CalculatorTool

model = OpenAIModel(model="gpt-3.5-turbo")
memory = AgentMemory(max_history=10)
tools = [CalculatorTool()]

agent = Agent(model=model, memory=memory, tools=tools, system_prompt="You are a helpful assistant.")
```

## 2. Run a Prompt

```python
response = agent.run("What is 123 * 456?")
print(response)
```

## 3. Use a Tool

The agent can use the calculator tool to answer math questions automatically.

---

See the [Quickstart](../getting-started/quickstart.md) for more examples. 