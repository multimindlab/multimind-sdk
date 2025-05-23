"""
Base Agent class for Multimind SDK.
"""

from typing import List, Dict, Any, Optional
from multimind.models.base import BaseLLM
from multimind.agents.memory import AgentMemory
from multimind.agents.tools.base import BaseTool

class Agent:
    """Base agent class that provides core agent functionality."""

    def __init__(
        self,
        model: BaseLLM,
        memory: Optional[AgentMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None
    ):
        self.model = model
        self.memory = memory or AgentMemory()
        self.tools = tools or []
        self.system_prompt = system_prompt

    async def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """Run the agent on a given task."""
        # Add task to memory
        self.memory.add_task(task)

        # Process task with tools and model
        response = await self._process_task(task, **kwargs)

        # Update memory with response
        self.memory.add_response(response)

        return response

    async def _process_task(self, task: str, **kwargs) -> Dict[str, Any]:
        """Process a task using available tools and the model."""
        # TODO: Implement task processing logic
        raise NotImplementedError("Task processing not implemented yet")

    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the agent."""
        self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent."""
        self.tools = [t for t in self.tools if t.name != tool_name]