import asyncio
from multimind.orchestration.dag_engine import DAGWorkflowEngine
from multimind.agents.thinker_agent import ThinkerAgent
from multimind.agents.self_reflect_agent import SelfReflectAgent
from multimind.memory.graph_memory import GraphMemoryAgent

# Mock GraphMemoryAgent for demonstration
class DummyGraphMemoryAgent:
    def query(self, predicate=None, obj=None):
        return [("A", "related_to", obj, {})]
    def add_fact(self, *args, **kwargs):
        pass

def simple_agent(name):
    def run(goal=None):
        return f"{name} completed with goal: {goal}"
    return run

async def main():
    dag = DAGWorkflowEngine()
    memory_agent = DummyGraphMemoryAgent()
    thinker = ThinkerAgent(memory_agent)
    self_reflector = SelfReflectAgent(thinker, memory_agent)

    # Add nodes: two parallel, one downstream
    dag.add_agent("think", thinker, dependencies=None)
    dag.add_agent("reflect", self_reflector, dependencies=["think"])
    dag.add_agent("simpleA", simple_agent("A"), dependencies=["think"])
    dag.add_agent("simpleB", simple_agent("B"), dependencies=["think"])
    dag.add_agent("join", simple_agent("Join"), dependencies=["simpleA", "simpleB", "reflect"])

    # Mutate DAG at runtime: add a new node after "join"
    def mutation_fn(graph):
        graph.add_node("final", agent=simple_agent("Final"))
        graph.add_edge("join", "final")
    dag.mutate_graph(mutation_fn)

    # Prepare input map for agents that require arguments
    input_map = {
        "think": {"goal": "demo-goal"},
        "reflect": {"goal": "demo-goal", "feedback": "improve"}
    }
    results = await dag.execute(input_map)
    for node, result in results.items():
        print(f"{node}: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 