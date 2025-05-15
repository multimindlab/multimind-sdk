# Architecture

MultiMind SDK is designed with modularity and extensibility in mind. The architecture enables seamless integration of models, agents, RAG, and orchestration workflows.

## Core Components
- **Pipeline:** Orchestrates model workflows, agent logic, and RAG operations.
- **Modules:** Model wrappers, agent system, RAG engine, CLI, logging, and integrations.
- **Utils:** Configuration management, environment loading, and utility functions.

## Flow Diagram

```mermaid
graph TD
    CLI[CLI] -->|commands| Pipeline
    API[Python API] -->|calls| Pipeline
    Pipeline -->|uses| Models
    Pipeline -->|manages| Agents
    Pipeline -->|invokes| RAG
    Pipeline -->|logs to| Logging
    Pipeline -->|integrates| Tools
    Models -->|wrappers| ModelWrappers
    Agents -->|memory/tools| AgentSystem
    RAG -->|vector store| VectorStore
    Logging -->|usage| UsageTracker
    Logging -->|trace| TraceLogger
```

This modular design allows for easy extension and integration with external tools and workflows. 