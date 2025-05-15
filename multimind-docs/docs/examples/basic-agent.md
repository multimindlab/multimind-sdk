# Basic Agent Example

This example demonstrates how to create and use a basic agent with MultiMind SDK.

```python
from multimind.models import OpenAIModel
from multimind.agents import Agent, AgentMemory
from multimind.agents.tools import CalculatorTool

model = OpenAIModel(model="gpt-3.5-turbo")
memory = AgentMemory(max_history=10)
tools = [CalculatorTool()]

agent = Agent(model=model, memory=memory, tools=tools, system_prompt="You are a helpful assistant.")
response = agent.run("What is 42 * 17?")
print(response)
``` 