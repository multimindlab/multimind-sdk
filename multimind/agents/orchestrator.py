"""
Bounded agent orchestration: a coordinator model decides, via a strict JSON
protocol, whether to answer directly, delegate to a registered agent, or spawn
a new sub-agent. Hard bounds (max_agents, max_depth, max_turns, budget) stop
spawning and force a direct answer — never silent truncation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from multimind.agents.agent import Agent
from multimind.agents.tools.base import BaseTool
from multimind.models.base import BaseLLM

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_ACTIONS = ("answer", "delegate", "spawn")
_REQUIRED_FIELDS = {
    "answer": ("content",),
    "delegate": ("agent", "task"),
    "spawn": ("name", "role", "task"),
}

_PROTOCOL = (
    "Decide the next step. Respond with exactly one JSON object and nothing else, "
    "in one of these forms:\n"
    '{"action": "answer", "content": "<final answer to the task>"}\n'
    '{"action": "delegate", "agent": "<registered agent name>", "task": "<subtask>"}\n'
    '{"action": "spawn", "name": "<new agent name>", "role": "<system prompt for the '
    'new agent>", "task": "<subtask>"}'
)


def parse_decision(text: str) -> Dict[str, Any]:
    """Parse a coordinator decision from model output; ValueError on garbage."""
    match = _JSON_RE.search(str(text))
    if not match:
        raise ValueError(f"Could not find a JSON object in model output: {text!r}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model output: {text!r}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got: {data!r}")
    action = data.get("action")
    if action not in _ACTIONS:
        raise ValueError(f"Unknown action {action!r} (use one of {_ACTIONS})")
    missing = [
        f
        for f in _REQUIRED_FIELDS[action]
        if not isinstance(data.get(f), str) or not data[f].strip()
    ]
    if missing:
        raise ValueError(f"Action {action!r} is missing required string fields: {missing}")
    return data


def _is_budget_error(exc: Exception) -> bool:
    # Lazy, optional coupling to multimind.observability.
    try:
        from multimind.observability.cost_tracker import BudgetExceededError
    except ImportError:
        BudgetExceededError = None
    if BudgetExceededError is not None and isinstance(exc, BudgetExceededError):
        return True
    return type(exc).__name__ == "BudgetExceededError"


@dataclass
class AgentSpec:
    """Declarative recipe for an agent the coordinator can delegate to."""

    name: str
    role: str
    model: Optional[Union[BaseLLM, Callable[[], BaseLLM]]] = None
    tools: List[BaseTool] = field(default_factory=list)
    max_turns: int = 5

    def resolve_model(self, default: BaseLLM) -> BaseLLM:
        if self.model is None:
            return default
        if isinstance(self.model, BaseLLM):
            return self.model
        if callable(self.model):
            return self.model()
        return self.model


@dataclass
class AgentNode:
    """One node in the agent tree: who ran, on what, and what came back."""

    name: str
    task: str
    depth: int
    result: Optional[str] = None
    children: List["AgentNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "depth": self.depth,
            "result": self.result,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class OrchestrationResult:
    """Outcome of one orchestrator run."""

    answer: str
    agent_tree: AgentNode
    turns_used: int
    bounds_hit: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "agent_tree": self.agent_tree.to_dict(),
            "turns_used": self.turns_used,
            "bounds_hit": list(self.bounds_hit),
        }


@dataclass
class _RunState:
    agents_used: int = 0
    turns: int = 0
    bounds_hit: List[str] = field(default_factory=list)


class AgentOrchestrator:
    """Coordinator loop with hard bounds on agents, depth, turns, and spend.

    Sub-agents with tools run as plain :class:`Agent` leaves; sub-agents
    without tools run the same decision loop and may spawn further sub-agents
    until a bound is hit. ``budget_tracker`` is any object with a ``check()``
    that raises ``BudgetExceededError`` (e.g. ``multimind.observability.Budget``).
    ``audit_hook`` receives a dict per lifecycle event and composes with the
    compliance ``AuditLog.write`` pattern.
    """

    def __init__(
        self,
        model: BaseLLM,
        specs: Optional[List[AgentSpec]] = None,
        max_agents: int = 10,
        max_depth: int = 2,
        budget_tracker: Optional[Any] = None,
        max_turns: int = 8,
        audit_hook: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.model = model
        self.max_agents = max_agents
        self.max_depth = max_depth
        self.max_turns = max_turns
        self.budget_tracker = budget_tracker
        self.audit_hook = audit_hook
        self.specs: Dict[str, AgentSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: AgentSpec) -> None:
        """Register an agent recipe the coordinator can delegate to."""
        if not isinstance(spec, AgentSpec):
            raise TypeError(f"Expected AgentSpec, got {type(spec).__name__}")
        self.specs[spec.name] = spec

    async def run(self, task: str) -> OrchestrationResult:
        """Run the coordinator loop on a task within the configured bounds."""
        state = _RunState()
        root = AgentNode(name="coordinator", task=task, depth=0)
        self._audit({"event": "run_started", "task": task})
        answer = await self._run_loop(task, None, root, 0, self.max_turns, self.model, state)
        root.result = answer
        self._audit(
            {"event": "run_completed", "turns": state.turns, "bounds_hit": list(state.bounds_hit)}
        )
        return OrchestrationResult(
            answer=answer,
            agent_tree=root,
            turns_used=state.turns,
            bounds_hit=state.bounds_hit,
        )

    async def _run_loop(
        self,
        task: str,
        role: Optional[str],
        node: AgentNode,
        depth: int,
        max_turns: int,
        model: BaseLLM,
        state: _RunState,
    ) -> str:
        observations: List[str] = []
        for _ in range(max_turns):
            state.turns += 1
            prompt = self._build_prompt(task, role, observations)
            try:
                decision = await self._decide(prompt, model)
            except Exception as e:
                if _is_budget_error(e):
                    self._record_bound("budget", node, state)
                    return self._fallback_answer(observations, "budget")
                raise
            action = decision["action"]
            if action == "answer":
                return decision["content"]
            # Delegating or spawning: enforce hard bounds first.
            if state.agents_used >= self.max_agents:
                self._record_bound("max_agents", node, state)
                return await self._force_answer(
                    task, role, observations, "max_agents", model, state
                )
            if depth >= self.max_depth:
                self._record_bound("max_depth", node, state)
                return await self._force_answer(task, role, observations, "max_depth", model, state)
            if action == "delegate":
                spec = self.specs.get(decision["agent"])
                if spec is None:
                    observations.append(
                        f"Delegation failed: no registered agent named {decision['agent']!r}. "
                        f"Registered agents: {sorted(self.specs)}"
                    )
                    continue
            else:
                spec = AgentSpec(
                    name=decision["name"], role=decision["role"], max_turns=self.max_turns
                )
            state.agents_used += 1
            child = AgentNode(name=spec.name, task=decision["task"], depth=depth + 1)
            node.children.append(child)
            self._audit(
                {
                    "event": "agent_spawned" if action == "spawn" else "agent_delegated",
                    "agent": spec.name,
                    "depth": depth + 1,
                    "task": decision["task"],
                }
            )
            result = await self._run_child(spec, decision["task"], child, depth + 1, state)
            child.result = result
            self._audit({"event": "agent_completed", "agent": spec.name, "depth": depth + 1})
            observations.append(f"Result from agent {spec.name!r}: {result}")
        self._record_bound("max_turns", node, state)
        return await self._force_answer(task, role, observations, "max_turns", model, state)

    async def _run_child(
        self, spec: AgentSpec, task: str, node: AgentNode, depth: int, state: _RunState
    ) -> str:
        model = spec.resolve_model(self.model)
        if spec.tools:
            agent = Agent(model=model, tools=list(spec.tools), system_prompt=spec.role)
            response = await agent.run(task)
            if "error" in response:
                return f"error: {response['error']}"
            return str(response.get("result"))
        return await self._run_loop(task, spec.role, node, depth, spec.max_turns, model, state)

    async def _decide(self, prompt: str, model: BaseLLM) -> Dict[str, Any]:
        self._check_budget()
        raw = await model.generate(prompt)
        try:
            return parse_decision(raw)
        except ValueError as e:
            retry_prompt = (
                f"{prompt}\n\nYour previous reply could not be parsed ({e}). "
                "Reply again with exactly one valid JSON object and nothing else."
            )
            self._check_budget()
            raw = await model.generate(retry_prompt)
            return parse_decision(raw)

    async def _force_answer(
        self,
        task: str,
        role: Optional[str],
        observations: List[str],
        reason: str,
        model: BaseLLM,
        state: _RunState,
    ) -> str:
        lines = [role] if role else []
        lines.append(f"Task: {task}")
        if observations:
            lines.append("Results so far:")
            lines.extend(f"- {o}" for o in observations)
        lines.append(
            f"The orchestration bound '{reason}' has been reached. You must now answer "
            "the task directly using the results so far. Do not delegate or spawn. "
            "Reply with your final answer."
        )
        prompt = "\n".join(lines)
        try:
            self._check_budget()
            raw = await model.generate(prompt)
        except Exception as e:
            if _is_budget_error(e):
                if "budget" not in state.bounds_hit:
                    state.bounds_hit.append("budget")
                return self._fallback_answer(observations, reason)
            raise
        try:
            decision = parse_decision(raw)
            if decision["action"] == "answer":
                return decision["content"]
        except ValueError:
            pass
        return str(raw).strip()

    def _fallback_answer(self, observations: List[str], reason: str) -> str:
        parts = [f"[orchestration stopped: bound '{reason}' reached before an answer was produced]"]
        parts.extend(observations)
        return "\n".join(parts)

    def _build_prompt(self, task: str, role: Optional[str], observations: List[str]) -> str:
        lines = [role] if role else []
        lines.append("You are coordinating work on the following task.")
        lines.append(f"Task: {task}")
        if self.specs:
            lines.append("Registered agents:")
            lines.extend(f"- {spec.name}: {spec.role}" for spec in self.specs.values())
        if observations:
            lines.append("Results so far:")
            lines.extend(f"- {o}" for o in observations)
        lines.append(_PROTOCOL)
        return "\n".join(lines)

    def _check_budget(self) -> None:
        if self.budget_tracker is None:
            return
        check = getattr(self.budget_tracker, "check", None)
        if callable(check):
            check()

    def _record_bound(self, bound: str, node: AgentNode, state: _RunState) -> None:
        state.bounds_hit.append(bound)
        logger.warning(
            "Orchestration bound %r hit at agent %r (depth %d); forcing direct answer",
            bound,
            node.name,
            node.depth,
        )
        self._audit({"event": "bound_hit", "bound": bound, "agent": node.name, "depth": node.depth})

    def _audit(self, event: Dict[str, Any]) -> None:
        if self.audit_hook is None:
            return
        try:
            self.audit_hook(dict(event))
        except Exception as e:
            logger.warning("Audit hook failed: %s", e)
