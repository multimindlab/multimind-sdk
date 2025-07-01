import matplotlib.pyplot as plt
import networkx as nx
from multimind.orchestration.dag_engine import DAGWorkflowEngine

def visualize_dag(dag: DAGWorkflowEngine, filename: str = None):
    G = dag.graph
    pos = nx.spring_layout(G)
    labels = {node: node for node in G.nodes}
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
    if filename:
        plt.savefig(filename)
    plt.show() 