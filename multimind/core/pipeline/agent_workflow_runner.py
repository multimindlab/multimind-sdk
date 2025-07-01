import networkx as nx
from typing import Dict, Any, List, Optional, Callable, Union
from multimind.agents.agent_loader import AgentLoader
from multimind.core.evolution.meta_controller_agent import MetaControllerAgent
from multimind.core.evolution.agent_mutator import AgentMutator
from multimind.core.evolution.agent_arena import AgentArena
from multimind.core.evolution.evolution_memory import EvolutionMemory
from multimind.core.evolution.multi_objective_judge_agent import MultiObjectiveJudgeAgent

class AgentWorkflowRunner:
    """
    Runs agents from a YAML/JSON-defined DAG (supports reflexion + mutation + evolutionary workflows).
    Loads agents, builds a directed acyclic graph (DAG), and executes the workflow.
    Supports runtime mutation (MetaControllerAgent, AgentMutator), reflexion/self-improvement hooks, and evolutionary workflows.

    Example usage:
        dag_config = {
            'nodes': [
                {'id': 'agent1', 'config': 'path/to/agent1.json'},
                {'id': 'agent2', 'config': 'path/to/agent2.json'}
            ],
            'edges': [
                {'from': 'agent1', 'to': 'agent2'}
            ]
        }
        runner = AgentWorkflowRunner(dag_config)
        output = await runner.run(input_data)

    Example evolutionary usage:
        dag_config = { ... }
        runner = AgentWorkflowRunner(dag_config, ...)
        await runner.run_evolutionary(input_data, rounds=3)
    """
    def __init__(self, dag_config: Dict[str, Any], agent_loader: Optional[AgentLoader] = None,
                 meta_controller: Optional[MetaControllerAgent] = None,
                 agent_mutator: Optional[AgentMutator] = None,
                 evolution_memory: Optional[EvolutionMemory] = None):
        """
        dag_config: Dict defining nodes (agents) and edges (data flow).
        agent_loader: Optional AgentLoader instance for loading agents from config.
        meta_controller: Optional MetaControllerAgent for runtime DAG mutation.
        agent_mutator: Optional AgentMutator for agent mutation logic.
        evolution_memory: Optional EvolutionMemory for recording evolutionary results.
        """
        self.dag_config = dag_config
        self.agent_loader = agent_loader or AgentLoader()
        self.meta_controller = meta_controller
        self.agent_mutator = agent_mutator
        self.evolution_memory = evolution_memory or EvolutionMemory()
        self.graph = nx.DiGraph()
        self.agents = {}
        self._build_dag()

    def _build_dag(self):
        """Builds the DAG and loads agents from config."""
        nodes = self.dag_config.get('nodes', [])
        edges = self.dag_config.get('edges', [])
        for node in nodes:
            agent_id = node['id']
            agent_cfg = node['config']
            # Support AgentArena as a node
            if isinstance(agent_cfg, dict) and agent_cfg.get('type') == 'arena':
                agents = [self.agent_loader.load_agent(cfg) for cfg in agent_cfg['agents']]
                judge = agent_cfg['judge']
                self.agents[agent_id] = AgentArena(agents, judge)
            else:
                agent = self.agent_loader.load_agent(agent_cfg)
                self.agents[agent_id] = agent
            self.graph.add_node(agent_id)
        for edge in edges:
            self.graph.add_edge(edge['from'], edge['to'])

    async def run(self, input_data: Any) -> Any:
        """
        Executes the DAG, passing data between agents according to the graph.
        Returns the output of the final node(s).
        Supports runtime mutation and reflexion hooks.
        """
        # Optionally mutate DAG before execution
        if self.meta_controller:
            self.meta_controller.mutate_dag(self.graph, self.agents)
        if self.agent_mutator:
            self.agent_mutator.mutate(self.graph, self.agents)

        # Find input nodes (no predecessors)
        input_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        data_map = {n: None for n in self.graph.nodes}
        for n in input_nodes:
            data_map[n] = input_data
        # Topological sort for execution order
        for node in nx.topological_sort(self.graph):
            agent = self.agents[node]
            # Gather inputs from predecessors
            preds = list(self.graph.predecessors(node))
            if preds:
                # For now, just take the output of the last predecessor
                input_val = data_map[preds[-1]]
            else:
                input_val = data_map[node]
            # Run agent (assume async call)
            if hasattr(agent, 'run') and callable(getattr(agent, 'run')):
                data_map[node] = await agent.run(input_val)
            else:
                data_map[node] = input_val  # Pass-through if no run method
        # Optionally trigger reflexion/self-improvement
        await self.self_reflect(data_map)
        # Return outputs from nodes with no successors
        output_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        return {n: data_map[n] for n in output_nodes}

    def mutate_dag(self):
        """
        Mutate the DAG structure at runtime using MetaControllerAgent or AgentMutator.
        """
        if self.meta_controller:
            self.meta_controller.mutate_dag(self.graph, self.agents)
        if self.agent_mutator:
            self.agent_mutator.mutate(self.graph, self.agents)

    async def self_reflect(self, data_map: Dict[str, Any]):
        """
        Reflexion/self-improvement hook after DAG execution.
        TODO: Integrate SelfReflectAgent or feedback loop here.
        """
        # Placeholder for reflexion logic
        pass

    async def run_evolutionary(self, input_data: Any, rounds: int = 3) -> Any:
        """
        Runs the workflow for several rounds, mutating agents and recording results in EvolutionMemory.
        Assumes the first node is an AgentArena.
        """
        node_ids = list(self.graph.nodes)
        if not node_ids:
            return None
        arena_node = node_ids[0]
        arena: AgentArena = self.agents[arena_node]
        results = []
        for i in range(rounds):
            print(f'=== Evolutionary Round {i+1} ===')
            result = await arena.run(input_data)
            print('Outputs:', result['outputs'])
            print('Best:', result['best'])
            self.evolution_memory.record(f'round_{i+1}', result['best'])
            results.append(result['best'])
            # Mutate agents for next round
            if self.meta_controller:
                self.meta_controller.mutate_dag(arena, {a.name: a for a in arena.agents})
            if self.agent_mutator:
                self.agent_mutator.mutate(arena, {a.name: a for a in arena.agents})
        print('\nEvolution Summary:')
        for i in range(rounds):
            summary = self.evolution_memory.summarize(f'round_{i+1}')
            print(f'Round {i+1} summary:', summary)
        return results

    # TODO: Add methods for runtime mutation (MetaControllerAgent, AgentMutator)
    # TODO: Add reflexion/self-improvement hooks 