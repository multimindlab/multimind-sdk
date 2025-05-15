# Quickstart

Get started with MultiMind SDK in minutes!

## 1. Create an Agent

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

## 2. Retrieval-Augmented Generation (RAG)

```python
from multimind.rag.rag import RAG
from multimind.rag.embeddings import OpenAIEmbedder
from multimind.rag.vector_store import FAISSVectorStore

embedder = OpenAIEmbedder()
vector_store = FAISSVectorStore()
rag = RAG(embedder=embedder, vector_store=vector_store)

rag.add_documents(["MultiMind SDK enables unified AI workflows."])
results = rag.query("What does MultiMind SDK do?")
print(results)
``` 