# Evolutionary Agentic AI Examples

This folder contains **modular, developer-friendly demos** for evolutionary and agentic workflows in MultiMindSDK.

## What You'll Find Here

- **agent_evolution_demo.py**: Run multiple agents in parallel, select the best with a multi-objective judge, and track performance over time.
- **agent_evolution_mutation_demo.py**: Adds mutation—evolve your agent pipeline between rounds and see how performance changes.
- **agent_workflow_demo.py**: Shows how to build and run a simple agent workflow (DAG) using the modular pipeline system.

## Quickstart

1. **Install dependencies** (from the project root):
   ```bash
   pip install -r requirements.txt
   ```
2. **Run a demo** (from the project root):
   ```bash
   python examples/evolutionary/agent_evolution_demo.py
   python examples/evolutionary/agent_evolution_mutation_demo.py
   python examples/evolutionary/agent_workflow_demo.py
   ```

## Modularity & Extensibility

- All examples use **modular agent classes**—swap in your own agents, judges, or memory modules.
- The architecture supports easy extension: add new agents, mutate pipelines, or plug in custom scoring and memory.
- Each script is self-contained but can be combined for more advanced workflows (e.g., hybrid symbolic + evolutionary).

## How to Extend

- **Add new agent strategies**: Subclass `MinimalAgent` or any agent and override the `run` method.
- **Customize judging**: Pass your own judge function or agent to the `AgentArena`.
- **Integrate memory or reasoning**: Use `GraphMemoryAgent`, `ThinkerAgent`, or `SelfReflectAgent` in your pipeline.
- **Experiment with mutation**: Use `MetaControllerAgent` and `AgentMutator` to evolve your agent pipeline.

## More Info

- See the main project docs for architectural details and advanced usage.
- All code is designed for clarity, modularity, and rapid prototyping.

---

Happy hacking! 🚀 