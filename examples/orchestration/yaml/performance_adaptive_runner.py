import asyncio
from multimind.orchestration.yaml_dag_parser import build_dag_from_yaml, register_agent

class MockAgent:
    def __init__(self, name):
        self.name = name
    def run(self, **kwargs):
        return f"{self.name} done"

register_agent('retriever', lambda: MockAgent('retriever'))
register_agent('llm', lambda: MockAgent('llm'))
register_agent('judge', lambda: MockAgent('judge'))
register_agent('fast_llm', lambda: MockAgent('fast_llm'))
register_agent('standard_llm', lambda: MockAgent('standard_llm'))

def main():
    dag = build_dag_from_yaml('examples/orchestration/yaml/performance_adaptive.yaml')
    results = asyncio.run(dag.execute())
    for node, result in results.items():
        print(f"{node}: {result}")

if __name__ == "__main__":
    main() 