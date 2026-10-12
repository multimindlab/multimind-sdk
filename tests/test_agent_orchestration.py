"""
Tests for bounded agent orchestration (multimind.agents.orchestrator).
"""

import json
from typing import Any, Dict, List, Optional, Union

import pytest

from multimind.agents.orchestrator import (
    AgentNode,
    AgentOrchestrator,
    AgentSpec,
    OrchestrationResult,
    parse_decision,
)
from multimind.agents.tools.base import BaseTool
from multimind.models.base import BaseLLM
from multimind.observability import Budget, BudgetExceededError


class ScriptedModel(BaseLLM):
    """Deterministic model that replays a scripted list of responses."""

    def __init__(self, responses: List[str], model_name: str = "scripted"):
        super().__init__(model_name)
        self.responses = list(responses)
        self.prompts: List[str] = []

    async def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs
    ):
        yield await self.generate(prompt)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        return await self.generate(messages[-1]["content"])

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        yield await self.chat(messages)

    async def embeddings(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, str):
            return [0.0]
        return [[0.0] for _ in text]


class EchoTool(BaseTool):
    def __init__(self):
        super().__init__(name="echo", description="Echoes a fixed value.")
        self.calls = 0

    async def run(self, **kwargs) -> Any:
        self.calls += 1
        return "echoed"

    def get_parameters(self) -> Dict[str, Any]:
        return {"required": []}


def answer(content: str) -> str:
    return json.dumps({"action": "answer", "content": content})


def delegate(agent: str, task: str) -> str:
    return json.dumps({"action": "delegate", "agent": agent, "task": task})


def spawn(name: str, role: str, task: str) -> str:
    return json.dumps({"action": "spawn", "name": name, "role": role, "task": task})


class TestParseDecision:
    def test_valid_answer(self):
        decision = parse_decision('{"action": "answer", "content": "42"}')
        assert decision == {"action": "answer", "content": "42"}

    def test_json_embedded_in_prose(self):
        decision = parse_decision('Sure! {"action": "answer", "content": "42"}')
        assert decision["content"] == "42"

    def test_no_json(self):
        with pytest.raises(ValueError):
            parse_decision("I think we should delegate this.")

    def test_invalid_json(self):
        with pytest.raises(ValueError):
            parse_decision('{"action": "answer", content: 42}')

    def test_unknown_action(self):
        with pytest.raises(ValueError):
            parse_decision('{"action": "explode"}')

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            parse_decision('{"action": "spawn", "name": "a"}')

    def test_non_string_field(self):
        with pytest.raises(ValueError):
            parse_decision('{"action": "answer", "content": 42}')


class TestDirectAnswer:
    async def test_answers_without_spawning(self):
        model = ScriptedModel([answer("Paris")])
        orch = AgentOrchestrator(model)
        result = await orch.run("What is the capital of France?")
        assert isinstance(result, OrchestrationResult)
        assert result.answer == "Paris"
        assert result.turns_used == 1
        assert result.bounds_hit == []
        assert result.agent_tree.name == "coordinator"
        assert result.agent_tree.children == []
        assert result.agent_tree.result == "Paris"


class TestDelegation:
    async def test_delegates_to_registered_agent(self):
        model = ScriptedModel(
            [
                delegate("researcher", "look up X"),
                answer("X is 7"),  # researcher's protocol loop answers
                answer("The value of X is 7"),
            ]
        )
        spec = AgentSpec(name="researcher", role="You research facts.")
        orch = AgentOrchestrator(model, specs=[spec])
        result = await orch.run("Find X")
        assert result.answer == "The value of X is 7"
        assert len(result.agent_tree.children) == 1
        child = result.agent_tree.children[0]
        assert child.name == "researcher"
        assert child.task == "look up X"
        assert child.result == "X is 7"
        # The sub-agent result reached the coordinator's next prompt.
        assert "X is 7" in model.prompts[-1]

    async def test_delegates_to_agent_with_tools(self):
        tool = EchoTool()
        model = ScriptedModel([delegate("helper", "please echo this"), answer("done: echoed")])
        spec = AgentSpec(name="helper", role="You use tools.", tools=[tool])
        orch = AgentOrchestrator(model, specs=[spec])
        result = await orch.run("Echo something")
        assert tool.calls == 1
        assert result.answer == "done: echoed"
        assert result.agent_tree.children[0].result == "echoed"

    async def test_unknown_agent_feeds_back_observation(self):
        model = ScriptedModel([delegate("ghost", "boo"), answer("no such agent")])
        orch = AgentOrchestrator(model)
        result = await orch.run("Try delegating")
        assert result.answer == "no such agent"
        assert result.agent_tree.children == []
        assert "ghost" in model.prompts[-1]


class TestSpawn:
    async def test_spawned_agent_runs_and_result_returns(self):
        model = ScriptedModel(
            [
                spawn("research", "You research.", "find facts about Y"),
                answer("Y facts found"),  # spawned agent's own loop
                answer("Summary: Y facts found"),
            ]
        )
        orch = AgentOrchestrator(model)
        result = await orch.run("Tell me about Y")
        assert result.answer == "Summary: Y facts found"
        assert result.bounds_hit == []
        child = result.agent_tree.children[0]
        assert child.name == "research"
        assert child.task == "find facts about Y"
        assert child.depth == 1
        assert child.result == "Y facts found"
        # Spawned agent saw its role and task; coordinator saw its result.
        assert any("You research." in p and "find facts about Y" in p for p in model.prompts)
        assert "Y facts found" in model.prompts[-1]

    async def test_nested_spawn_within_depth(self):
        model = ScriptedModel(
            [
                spawn("a", "role a", "task a"),
                spawn("b", "role b", "task b"),  # agent a spawns b (depth 2 allowed)
                answer("b done"),
                answer("a done via b"),
                answer("final"),
            ]
        )
        orch = AgentOrchestrator(model, max_depth=2)
        result = await orch.run("nest")
        assert result.answer == "final"
        assert result.bounds_hit == []
        a = result.agent_tree.children[0]
        b = a.children[0]
        assert (a.name, a.depth, b.name, b.depth) == ("a", 1, "b", 2)
        assert b.result == "b done"


class TestBounds:
    async def test_max_agents_stops_spawning_and_forces_answer(self):
        model = ScriptedModel(
            [
                spawn("one", "role", "task one"),
                answer("one done"),
                spawn("two", "role", "task two"),  # refused: max_agents=1
                "forced final answer",  # forced direct answer
            ]
        )
        orch = AgentOrchestrator(model, max_agents=1)
        result = await orch.run("do lots")
        assert "max_agents" in result.bounds_hit
        assert result.answer == "forced final answer"
        assert len(result.agent_tree.children) == 1
        assert "one done" in model.prompts[-1]

    async def test_max_depth_stops_nested_spawn(self):
        model = ScriptedModel(
            [
                spawn("child", "role", "child task"),
                spawn("grandchild", "role", "too deep"),  # refused: depth 1 == max_depth
                answer("child forced answer"),  # forced answer accepts JSON answer too
                answer("final"),
            ]
        )
        orch = AgentOrchestrator(model, max_depth=1)
        result = await orch.run("go deep")
        assert "max_depth" in result.bounds_hit
        assert result.answer == "final"
        child = result.agent_tree.children[0]
        assert child.children == []
        assert child.result == "child forced answer"

    async def test_max_turns_forces_answer(self):
        model = ScriptedModel(
            [
                spawn("a", "role", "task a"),
                answer("a done"),
                "ran out of turns",  # coordinator forced after 1 turn
            ]
        )
        orch = AgentOrchestrator(model, max_turns=1)
        result = await orch.run("slow task")
        assert "max_turns" in result.bounds_hit
        assert result.answer == "ran out of turns"

    async def test_budget_exhausted_blocks_calls(self):
        budget = Budget(max_cost=0.01)
        budget.add(0.02)
        with pytest.raises(BudgetExceededError):
            budget.check()
        model = ScriptedModel([answer("never used")])
        orch = AgentOrchestrator(model, budget_tracker=budget)
        result = await orch.run("anything")
        assert result.bounds_hit == ["budget"]
        assert "budget" in result.answer
        assert model.prompts == []  # no model call was dispatched


class TestMalformedJSON:
    async def test_retry_then_success(self):
        model = ScriptedModel(["not json at all", answer("recovered")])
        orch = AgentOrchestrator(model)
        result = await orch.run("task")
        assert result.answer == "recovered"
        assert len(model.prompts) == 2
        assert "could not be parsed" in model.prompts[1]

    async def test_retry_then_valueerror(self):
        model = ScriptedModel(["garbage", "still garbage"])
        orch = AgentOrchestrator(model)
        with pytest.raises(ValueError):
            await orch.run("task")
        assert len(model.prompts) == 2


class TestAuditHook:
    async def test_lifecycle_events_emitted(self):
        events: List[Dict[str, Any]] = []
        model = ScriptedModel(
            [
                spawn("worker", "role", "subtask"),
                answer("worker done"),
                answer("final"),
            ]
        )
        orch = AgentOrchestrator(model, audit_hook=events.append)
        await orch.run("task")
        names = [e["event"] for e in events]
        assert names == ["run_started", "agent_spawned", "agent_completed", "run_completed"]
        spawned = events[1]
        assert spawned["agent"] == "worker"
        assert spawned["depth"] == 1
        assert spawned["task"] == "subtask"

    async def test_bound_hit_event_and_failing_hook_is_swallowed(self):
        events: List[Dict[str, Any]] = []

        def hook(event):
            events.append(event)
            raise RuntimeError("hook boom")

        model = ScriptedModel(
            [
                spawn("one", "role", "t1"),
                answer("done"),
                spawn("two", "role", "t2"),
                "forced",
            ]
        )
        orch = AgentOrchestrator(model, max_agents=1, audit_hook=hook)
        result = await orch.run("task")
        assert result.answer == "forced"
        assert any(e["event"] == "bound_hit" and e["bound"] == "max_agents" for e in events)


class TestResultStructure:
    async def test_agent_tree_to_dict(self):
        model = ScriptedModel(
            [
                spawn("research", "You research.", "find"),
                answer("found"),
                answer("final"),
            ]
        )
        orch = AgentOrchestrator(model)
        result = await orch.run("task")
        tree = result.agent_tree.to_dict()
        assert tree["name"] == "coordinator"
        assert tree["depth"] == 0
        assert tree["result"] == "final"
        assert tree["children"][0] == {
            "name": "research",
            "task": "find",
            "depth": 1,
            "result": "found",
            "children": [],
        }
        as_dict = result.to_dict()
        assert as_dict["answer"] == "final"
        assert as_dict["turns_used"] == result.turns_used
        assert as_dict["bounds_hit"] == []


class TestConstruction:
    def test_invalid_bounds_rejected(self):
        model = ScriptedModel([])
        with pytest.raises(ValueError):
            AgentOrchestrator(model, max_agents=0)
        with pytest.raises(ValueError):
            AgentOrchestrator(model, max_depth=0)
        with pytest.raises(ValueError):
            AgentOrchestrator(model, max_turns=0)

    def test_register_rejects_non_spec(self):
        orch = AgentOrchestrator(ScriptedModel([]))
        with pytest.raises(TypeError):
            orch.register({"name": "nope"})

    def test_spec_model_factory_resolution(self):
        default = ScriptedModel([], model_name="default")
        other = ScriptedModel([], model_name="other")
        assert AgentSpec(name="a", role="r").resolve_model(default) is default
        assert AgentSpec(name="a", role="r", model=other).resolve_model(default) is other
        assert AgentSpec(name="a", role="r", model=lambda: other).resolve_model(default) is other

    def test_exports(self):
        from multimind.agents import (
            AgentNode as N,
        )
        from multimind.agents import (
            AgentOrchestrator as O,
        )
        from multimind.agents import (
            AgentSpec as S,
        )
        from multimind.agents import (
            OrchestrationResult as R,
        )

        assert (N, O, S, R) == (AgentNode, AgentOrchestrator, AgentSpec, OrchestrationResult)
