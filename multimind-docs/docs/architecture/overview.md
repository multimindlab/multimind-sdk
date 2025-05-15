# Architecture Overview

MultiMind SDK is designed for modularity, extensibility, and developer productivity. The architecture enables seamless integration of models, agents, RAG, and orchestration workflows.

## High-Level Diagram

```mermaid
graph TD
    subgraph User
        CLI[CLI]
        API[Python API]
    end
    subgraph Core
        Agent[Agent System]
        MCP[MCP Engine]
        Router[Model Router]
        Chain[Prompt Chain]
        Runner[Task Runner]
    end
    subgraph Models
        OpenAI[OpenAI]
        Claude[Claude]
        Mistral[Mistral]
        HF[HuggingFace]
        Ollama[Ollama]
    end
    subgraph Tools
        Memory[Agent Memory]
        Calculator[Calculator]
        WebSearch[Web Search]
        FileOps[File Operations]
    end
    subgraph Monitoring
        Usage[Usage Tracker]
        Trace[Trace Logger]
        Export[Export/Report]
    end
    CLI -->|commands| Agent
    CLI -->|workflows| MCP
    API -->|calls| Agent
    API -->|executes| MCP
    Agent -->|uses| Router
    Agent -->|manages| Memory
    Agent -->|invokes| Tools
    MCP -->|composes| Models
    MCP -->|orchestrates| Chain
    MCP -->|runs| Runner
    Router -->|routes to| Models
    Chain -->|uses| Models
    Runner -->|executes| Chain
    Agent -->|logs to| Usage
    MCP -->|logs to| Trace
    Usage -->|exports to| Export
    Trace -->|exports to| Export
```

## Main Components
- **Agent System:** Manages agent logic, memory, and tool use.
- **RAG Engine:** Handles document ingestion, embedding, and retrieval.
- **Model Router:** Selects the best model for each task.
- **Orchestration:** Enables prompt chaining and workflow execution.
- **Monitoring:** Tracks usage, logs traces, and exports reports.

For more details, see the [Features](../features/core-features.md) and [API Reference](../api/rag-api.md). 