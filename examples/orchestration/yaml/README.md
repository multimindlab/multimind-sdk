# YAML-to-DAG Orchestration Examples

This folder demonstrates how to define and execute agent workflows as Directed Acyclic Graphs (DAGs) using YAML configuration files and the MultiMindSDK orchestration engine.

## 🧩 How It Works
- **Pipelines** are defined declaratively in YAML, specifying agents, branches, and conditions.
- The `build_dag_from_yaml` function parses the YAML and constructs a `DAGWorkflowEngine`.
- Each runner script registers mock agents, loads the YAML, and executes the workflow.

## 🚀 Quickstart

1. **Register your agents** in the runner script using `register_agent('name', constructor)`.
2. **Run the example**:
   ```bash
   python basic_linear_runner.py
   python branching_runner.py
   python conditional_router_runner.py
   python performance_adaptive_runner.py
   python symbolic_evolution_runner.py
   ```
3. **See the results** printed for each node in the pipeline.

## 📄 Example Pipelines

| YAML File                  | Description                                      |
|---------------------------|--------------------------------------------------|
| basic_linear.yaml          | Simple linear pipeline: retriever → llm → judge  |
| branching.yaml             | Parallel branches after retriever                |
| conditional_router.yaml    | Conditional subflow based on judge.score         |
| performance_adaptive.yaml  | Branches based on judge.latency (performance)    |
| symbolic_evolution.yaml    | Symbolic evolution: insert improver if needed    |

## 🛠️ How to Register Agents

In each runner, register your agent classes or functions:
```python
register_agent('retriever', lambda: RetrieverAgent())
register_agent('llm', lambda: LLM())
# ...
```

## 📝 Extending
- Add new YAML files for more complex workflows.
- Implement real agent logic and register them in the runners.
- Use the `MetaControllerAgent` for runtime graph mutation and adaptation.

## 📚 References
- See `multimind/orchestration/yaml_dag_parser.py` for the parser implementation.
- See `multimind/orchestration/dag_engine.py` for the DAG execution engine. 