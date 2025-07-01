import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from multimind.agents.agent import Agent
from multimind.agents.memory import AgentMemory
from multimind.agents.tools.base import BaseTool
from multimind.models.base import BaseLLM

class AgentLoader:
    """
    Loads agent configurations from Python dict, JSON, or YAML files.
    Supports developer-friendly modular agent loading for pipelines and workflows.
    """
    def __init__(self, model_registry: Optional[Dict[str, BaseLLM]] = None):
        self.model_registry = model_registry or {}
        self.tool_registry: Dict[str, BaseTool] = {}

    def register_model(self, name: str, model: BaseLLM) -> None:
        self.model_registry[name] = model

    def register_tool(self, name: str, tool: BaseTool) -> None:
        self.tool_registry[name] = tool

    def load_agent(
        self,
        config_path_or_dict: Union[str, Dict[str, Any]],
        model: Optional[BaseLLM] = None,
        tools: Optional[List[BaseTool]] = None
    ) -> Agent:
        """
        Load an agent from a configuration file (YAML/JSON) or dict.
        """
        # If config is a dict with 'class', instantiate directly (for demo)
        if isinstance(config_path_or_dict, dict) and 'class' in config_path_or_dict:
            agent_class = config_path_or_dict['class']
            kwargs = config_path_or_dict.get('kwargs', {})
            return agent_class(**kwargs)
        # If config is a dict, treat as config
        if isinstance(config_path_or_dict, dict):
            config = config_path_or_dict
        else:
            # Load from file (YAML or JSON)
            path = Path(config_path_or_dict)
            if not path.exists():
                raise FileNotFoundError(f"Agent config file not found: {config_path_or_dict}")
            if path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config support.")
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file type: {path.suffix}")
        # Validate config
        required_keys = {"model", "system_prompt"}
        if not all(key in config for key in required_keys):
            raise ValueError(f"Agent config must contain: {required_keys}")
        # Get or create model
        if model is None:
            model_name = config["model"]
            if model_name not in self.model_registry:
                raise ValueError(f"Model not registered: {model_name}")
            model = self.model_registry[model_name]
        # Get or create tools
        if tools is None:
            tools = []
            for tool_name in config.get("tools", []):
                if tool_name not in self.tool_registry:
                    raise ValueError(f"Tool not registered: {tool_name}")
                tools.append(self.tool_registry[tool_name])
        # Create memory
        memory_config = config.get("memory", {})
        memory = AgentMemory(
            max_history=memory_config.get("max_history", 100)
        )
        # Create agent
        agent = Agent(
            model=model,
            memory=memory,
            tools=tools,
            system_prompt=config["system_prompt"]
        )
        return agent 