# Integration Guide

This guide demonstrates how to integrate MultiMind SDK into your own Python projects or APIs.

## Installation
Install the SDK in your project environment:
```
pip install multimind-sdk
```

## Example: Using MultiMind in a Python Script
```python
from multimind.models import OpenAIModel
from multimind.agents import Agent, AgentMemory
from multimind.agents.tools import CalculatorTool

# Initialize model and agent
model = OpenAIModel(model="gpt-3.5-turbo")
memory = AgentMemory(max_history=10)
tools = [CalculatorTool()]
agent = Agent(model=model, memory=memory, tools=tools, system_prompt="You are a helpful assistant.")

# Use the agent in your application
response = agent.run("What is 42 * 17?")
print(response)
```

## Example: Integrating with a FastAPI Endpoint
```python
from fastapi import FastAPI
from multimind.models import OpenAIModel
from multimind.agents import Agent, AgentMemory

app = FastAPI()
model = OpenAIModel(model="gpt-3.5-turbo")
memory = AgentMemory(max_history=10)
agent = Agent(model=model, memory=memory, tools=[], system_prompt="You are a helpful assistant.")

@app.post("/ask")
def ask(prompt: str):
    return {"response": agent.run(prompt)}
```

You can expand this pattern to integrate MultiMind SDK with any Python-based service or workflow. 