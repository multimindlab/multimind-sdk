import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio

class DAGWorkflowEngine:
    """
    Orchestrates agent workflows as a directed acyclic graph (DAG).
    Supports dynamic mutation and parallel execution of independent nodes.
    Each node is an agent (callable or object with a .run() method).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_results: Dict[str, Any] = {}

    def add_agent(self, name: str, agent: Any, dependencies: Optional[List[str]] = None):
        """Add an agent node to the DAG with optional dependencies (edges)."""
        self.graph.add_node(name, agent=agent)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, name)

    def remove_agent(self, name: str):
        """Remove an agent node and its edges from the DAG."""
        self.graph.remove_node(name)
        if name in self.node_results:
            del self.node_results[name]

    def update_dependencies(self, name: str, new_dependencies: List[str]):
        """Update dependencies (edges) for a given node."""
        self.graph.remove_edges_from(list(self.graph.in_edges(name)))
        for dep in new_dependencies:
            self.graph.add_edge(dep, name)

    def get_ready_nodes(self, completed: Set[str]) -> List[str]:
        """Return nodes whose dependencies are all satisfied and not yet run."""
        ready = []
        for node in self.graph.nodes:
            if node in completed:
                continue
            preds = set(self.graph.predecessors(node))
            if preds.issubset(completed):
                ready.append(node)
        return ready

    async def _run_agent(self, name: str, agent: Any, inputs: Dict[str, Any]) -> Any:
        if hasattr(agent, 'run'):
            return await asyncio.to_thread(agent.run, **inputs)
        elif callable(agent):
            return await asyncio.to_thread(agent, **inputs)
        else:
            raise ValueError(f"Agent {name} is not callable or does not have a .run() method.")

    async def execute(self, input_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the DAG, running agents in parallel where possible.
        input_map: Optional dict mapping node names to input kwargs.
        Returns a dict of node results.
        """
        input_map = input_map or {}
        completed: Set[str] = set()
        pending: Set[str] = set()
        while len(completed) < len(self.graph.nodes):
            ready = self.get_ready_nodes(completed)
            if not ready:
                raise RuntimeError("Cyclic dependency or no runnable nodes left.")
            tasks = []
            for node in ready:
                agent = self.graph.nodes[node]['agent']
                inputs = input_map.get(node, {})
                tasks.append((node, self._run_agent(node, agent, inputs)))
                pending.add(node)
            results = await asyncio.gather(*(t[1] for t in tasks))
            for idx, (node, _) in enumerate(tasks):
                self.node_results[node] = results[idx]
                completed.add(node)
        return self.node_results

    def mutate_graph(self, mutation_fn: Callable[[nx.DiGraph], None]):
        """Apply a mutation function to the underlying graph (for MetaControllerAgent, etc)."""
        mutation_fn(self.graph) 