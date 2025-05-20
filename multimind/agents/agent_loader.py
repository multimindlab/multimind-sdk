"""
Agent loader for loading agent configurations from MCP files.
"""

import json
from typing import Dict, Any, Optional, Lis
from pathlib import Path
from multimind.agents.agent import Agen
from multimind.agents.memory import AgentMemory
from multimind.agents.tools.base import BaseTool
from multimind.models.base import BaseLLM

class AgentLoader:
    """Loads agent configurations from MCP files."""

    def __init__(self, model_registry: Optional[Dict[str, BaseLLM]] = None):
        self.model_registry = model_registry or {}
        self.tool_registry: Dict[str, BaseTool] = {}

    def register_model(self, name: str, model: BaseLLM) -> None:
        """Register a model for use in agents."""
        self.model_registry[name] = model

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Register a tool for use in agents."""
        self.tool_registry[name] = tool

    def load_agent(
        self,
        config_path: str,
        model: Optional[BaseLLM] = None,
        tools: Optional[List[BaseTool]] = None
    ) -> Agent:
        """Load an agent from a configuration file."""
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

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

        # Create agen
        agent = Agent(
            model=model,
            memory=memory,
            tools=tools,
            system_prompt=config["system_prompt"]
        )

        return agen

    def load_agents_from_dir(
        self,
        dir_path: str,
        model: Optional[BaseLLM] = None
    ) -> Dict[str, Agent]:
        """Load multiple agents from a directory of config files."""
        agents = {}
        config_dir = Path(dir_path)

        for config_file in config_dir.glob("*.json"):
            agent_name = config_file.stem
            agents[agent_name] = self.load_agent(
                str(config_file),
                model=model
            )

        return agents