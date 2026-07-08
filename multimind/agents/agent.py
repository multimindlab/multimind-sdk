"""
Base Agent class for Multimind SDK.
"""

import json
import re
from typing import Any, Dict, List, Optional

from multimind.agents.memory import AgentMemory
from multimind.agents.tools.base import BaseTool
from multimind.models.base import BaseLLM


def _supports_native_tools(model: Any) -> bool:
    """Duck-typed check for an OpenAI-compatible chat-completions client.

    Native tool-calling needs access to the raw assistant message (to read
    ``tool_calls``), which the string-returning ``BaseLLM.chat`` API does not
    expose. Any model that publishes an OpenAI-shaped async client at
    ``model.client.chat.completions.create`` (OpenAIModel and every
    OpenAI-compatible provider in this SDK) qualifies. Everything else falls
    back to keyword-based tool routing.
    """
    create = getattr(
        getattr(getattr(getattr(model, "client", None), "chat", None), "completions", None),
        "create",
        None,
    )
    return callable(create)


def build_tool_schema(tool: BaseTool) -> Dict[str, Any]:
    """Build an OpenAI function-tool definition from a BaseTool.

    Uses an explicit ``tool.schema`` attribute when present, otherwise derives
    the JSON schema from ``tool.get_parameters()`` (``properties``/``required``).
    """
    explicit = getattr(tool, "schema", None)
    if isinstance(explicit, dict):
        parameters = explicit
    else:
        params = tool.get_parameters() or {}
        parameters = {
            "type": "object",
            "properties": params.get("properties", {}),
            "required": params.get("required", []),
        }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


class Agent:
    """Base agent class that provides core agent functionality.

    When the underlying model exposes an OpenAI-compatible client, tools are
    offered to the model as native function definitions and executed from the
    returned ``tool_calls`` in a bounded loop. Otherwise the agent falls back
    to keyword-based tool routing.
    """

    def __init__(
        self,
        model: BaseLLM,
        memory: Optional[AgentMemory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 5,
    ):
        self.model = model
        self.memory = memory or AgentMemory()
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.max_turns = max_turns

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
        if self.tools and _supports_native_tools(self.model):
            return await self._run_tool_loop(task, **kwargs)
        return await self._process_task_keyword(task, **kwargs)

    async def _run_tool_loop(self, task: str, **kwargs) -> Dict[str, Any]:
        """Native function-calling loop against an OpenAI-compatible client.

        Sends JSON-schema tool definitions, executes every returned tool call,
        feeds results back as ``tool`` messages, and stops when the model
        answers without tool calls or ``max_turns`` is exhausted. Tool and
        argument errors are reported back to the model instead of raising, so
        a malformed call gets a chance to be corrected.
        """
        max_turns = kwargs.pop("max_turns", self.max_turns)
        tool_defs = [build_tool_schema(tool) for tool in self.tools]
        tool_map = {tool.name: tool for tool in self.tools}

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})

        executed: List[Dict[str, Any]] = []
        try:
            for _ in range(max_turns):
                response = await self.model.client.chat.completions.create(
                    model=self.model.model_name,
                    messages=messages,
                    tools=tool_defs,
                    **kwargs,
                )
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    return {
                        "type": "model",
                        "result": message.content or "",
                        "tool_calls": executed,
                    }

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                for call in tool_calls:
                    record, content = await self._execute_tool_call(call, tool_map)
                    executed.append(record)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            return {
                "type": "model",
                "error": f"max_turns ({max_turns}) exhausted without a final answer",
                "tool_calls": executed,
            }
        except Exception as e:
            return {"type": "model", "error": str(e), "tool_calls": executed}

    @staticmethod
    async def _execute_tool_call(call: Any, tool_map: Dict[str, BaseTool]):
        """Execute one tool call; never raises. Returns (record, message_content)."""
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError(f"tool arguments must be a JSON object, got {type(args).__name__}")
        except (json.JSONDecodeError, ValueError) as e:
            error = f"Malformed arguments for tool '{name}': {e}"
            return {"tool": name, "args": call.function.arguments, "error": error}, error

        tool = tool_map.get(name)
        if tool is None:
            error = f"Unknown tool '{name}' (available: {', '.join(sorted(tool_map))})"
            return {"tool": name, "args": args, "error": error}, error

        if not tool.validate_parameters(**args):
            required = tool.get_parameters().get("required", [])
            error = (
                f"Missing required parameters for tool '{name}': "
                f"expected {required}, got {sorted(args)}"
            )
            return {"tool": name, "args": args, "error": error}, error

        try:
            result = await tool.run(**args)
        except Exception as e:
            error = f"Tool '{name}' failed: {e}"
            return {"tool": name, "args": args, "error": error}, error
        try:
            content = json.dumps(result, default=str)
        except (TypeError, ValueError):
            content = str(result)
        return {"tool": name, "args": args, "result": result}, content

    async def _process_task_keyword(self, task: str, **kwargs) -> Dict[str, Any]:
        """Keyword-routing fallback for models without native tool-calling."""
        # Try to match a tool by name
        task_lower = task.lower()
        for tool in self.tools:
            tool_name = tool.name.lower().strip()
            # Match on whole tokens only (e.g., `calc` won't match `calculate`).
            if re.search(rf"\b{re.escape(tool_name)}\b", task_lower):
                try:
                    # Extract parameters for the tool from kwargs
                    params = {
                        k: v
                        for k, v in kwargs.items()
                        if k in tool.get_parameters().get("required", [])
                    }
                    if not tool.validate_parameters(**params):
                        raise ValueError(f"Missing required parameters for tool '{tool.name}'")
                    result = await tool.run(**params)
                    return {"type": "tool", "tool": tool.name, "result": result}
                except Exception as e:
                    return {"type": "tool", "tool": tool.name, "error": str(e)}
        # If no tool matches, use the model
        try:
            prompt = task
            model_result = await self.model.generate(prompt, **kwargs)
            return {"type": "model", "result": model_result}
        except Exception as e:
            return {"type": "model", "error": str(e)}

    def add_tool(self, tool: BaseTool) -> None:
        """Add a new tool to the agent."""
        self.tools.append(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the agent."""
        self.tools = [t for t in self.tools if t.name != tool_name]
