import asyncio
from multimind.orchestration.yaml_dag_parser import build_dag_from_yaml, register_agent
from multimind.agents.thinker_agent import ThinkerAgent
from multimind.agents.self_reflect_agent import SelfReflectAgent

# Dummy memory agent for demonstration
class DummyGraphMemoryAgent:
    def query(self, predicate=None, obj=None):
        return [("A", "related_to", obj, {})]
    def add_fact(self, *args, **kwargs):
        pass

memory_agent = DummyGraphMemoryAgent()
thinker = lambda: ThinkerAgent(memory_agent)
self_reflector = lambda: SelfReflectAgent(ThinkerAgent(memory_agent), memory_agent)

register_agent('retriever', thinker)
register_agent('llm', self_reflector)
register_agent('judge', thinker)

def main():
    dag = build_dag_from_yaml('examples/orchestration/yaml/basic_linear.yaml')
    # Provide input for agents that require it
    input_map = {
        'retriever': {'goal': 'demo-goal'},
        'llm': {'goal': 'demo-goal', 'feedback': 'improve'},
        'judge': {'goal': 'demo-goal'}
    }
    results = asyncio.run(dag.execute(input_map))
    for node, result in results.items():
        print(f"{node}: {result}")

if __name__ == "__main__":
    main() 