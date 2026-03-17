import sys
from multimind.orchestration.yaml_dag_parser import build_dag_from_yaml, register_agent
from multimind.orchestration.dag_visualization import visualize_dag

# Register mock agents for demonstration
class MockAgent:
    def __init__(self, name):
        self.name = name
    def run(self, **kwargs):
        return f"{self.name} done"

for name in [
    'retriever', 'llm', 'judge', 'summarizer', 'rewriter', 'fast_llm', 'standard_llm', 'improver']:
    register_agent(name, lambda n=name: MockAgent(n))

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_yaml_dag.py <yaml_file>")
        sys.exit(1)
    yaml_file = sys.argv[1]
    dag = build_dag_from_yaml(yaml_file)
    visualize_dag(dag)

if __name__ == "__main__":
    main() 