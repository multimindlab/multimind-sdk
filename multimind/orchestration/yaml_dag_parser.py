import yaml
from multimind.orchestration.dag_engine import DAGWorkflowEngine
from typing import Any, Dict

# Registry for agent constructors (to be extended as needed)
AGENT_REGISTRY = {}

def register_agent(name: str, constructor):
    AGENT_REGISTRY[name] = constructor

def build_dag_from_yaml(yaml_path: str) -> DAGWorkflowEngine:
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    dag = DAGWorkflowEngine()
    node_params = {}
    def add_node(node, dependencies=None):
        if isinstance(node, dict):
            name = node['name']
            params = node.get('params', {})
        else:
            name = node
            params = {}
        agent_ctor = AGENT_REGISTRY.get(name)
        if agent_ctor is None:
            raise ValueError(f"Agent '{name}' not registered.")
        dag.add_agent(name, agent_ctor(), dependencies)
        node_params[name] = params
    def walk_pipeline(pipeline, prev=None):
        if isinstance(pipeline, list):
            last = prev
            for node in pipeline:
                if isinstance(node, dict) and any(k in node for k in ('if', 'switch', 'fallback')):
                    # Handle router/conditional/fallback
                    for k in ('if', 'switch', 'fallback'):
                        if k in node:
                            cond = node[k]
                            cond_name = f"{k}_{len(dag.graph.nodes)}"
                            dag.add_agent(cond_name, lambda: None, [last] if last else None)
                            for branch in cond['then']:
                                walk_pipeline(branch, cond_name)
                            last = cond_name
                else:
                    add_node(node, [last] if last else None)
                    last = node['name'] if isinstance(node, dict) else node
        else:
            add_node(pipeline, [prev] if prev else None)
    walk_pipeline(config['pipeline'])
    dag.node_params = node_params
    return dag 