from typing import Callable, Any
import networkx as nx

class MetaControllerAgent:
    """
    Observes DAGWorkflowEngine execution and mutates the DAG dynamically.
    Can insert, remove, or reorder nodes/edges at runtime based on feedback or results.
    """
    def __init__(self, mutation_policy: Callable[[nx.DiGraph, dict], None]):
        """
        mutation_policy: function that takes (graph, results) and mutates the graph in-place
        """
        self.mutation_policy = mutation_policy

    def observe_and_mutate(self, dag_engine: Any, results: dict):
        """
        Observe the current DAG and results, then mutate the DAG in-place.
        dag_engine: instance of DAGWorkflowEngine
        results: dict of node results so far
        """
        self.mutation_policy(dag_engine.graph, results)

    @staticmethod
    def insert_node_if_metric_high(node_name: str, metric_key: str, threshold: float, new_node: str, agent_ctor):
        """
        Returns a mutation policy that inserts new_node after node_name if results[node_name][metric_key] > threshold.
        """
        def policy(graph: nx.DiGraph, results: dict):
            if node_name in results and results[node_name].get(metric_key, 0) > threshold:
                # Insert new_node after node_name
                succs = list(graph.successors(node_name))
                graph.add_node(new_node, agent=agent_ctor())
                for succ in succs:
                    graph.remove_edge(node_name, succ)
                    graph.add_edge(new_node, succ)
                graph.add_edge(node_name, new_node)
        return policy

    @staticmethod
    def remove_node_if_metric_low(node_name: str, metric_key: str, threshold: float):
        """
        Returns a mutation policy that removes node_name if results[node_name][metric_key] < threshold.
        """
        def policy(graph: nx.DiGraph, results: dict):
            if node_name in results and results[node_name].get(metric_key, 0) < threshold:
                preds = list(graph.predecessors(node_name))
                succs = list(graph.successors(node_name))
                graph.remove_node(node_name)
                for p in preds:
                    for s in succs:
                        graph.add_edge(p, s)
        return policy

    @staticmethod
    def replace_node(node_name: str, new_node: str, agent_ctor):
        """
        Returns a mutation policy that replaces node_name with new_node.
        """
        def policy(graph: nx.DiGraph, results: dict):
            if node_name in graph.nodes:
                preds = list(graph.predecessors(node_name))
                succs = list(graph.successors(node_name))
                graph.remove_node(node_name)
                graph.add_node(new_node, agent=agent_ctor())
                for p in preds:
                    graph.add_edge(p, new_node)
                for s in succs:
                    graph.add_edge(new_node, s)
        return policy

    # Example usage:
    # policy = MetaControllerAgent.insert_node_if_metric_high('judge', 'latency', 2.0, 'fast_llm', FastLLMAgent)
    # meta_agent = MetaControllerAgent(policy)
    # meta_agent.observe_and_mutate(dag_engine, results) 