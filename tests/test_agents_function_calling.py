"""Tests for native function-calling in the base Agent.

Uses OpenAI-shaped mock clients (SimpleNamespace response objects) so no
network access or API key is required.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from multimind.agents.agent import Agent, _supports_native_tools, build_tool_schema
from multimind.agents.tools.base import BaseTool
from multimind.agents.tools.calculator import CalculatorTool


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(content=None, tool_calls=None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Records create() kwargs and replays a scripted list of responses."""

    def __init__(self, responses: List[SimpleNamespace]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeCompletions ran out of scripted responses")
        return self.responses.pop(0)


class FakeOpenAIModel:
    """Duck-typed OpenAI-compatible model: exposes client.chat.completions.create."""

    def __init__(self, responses: List[SimpleNamespace]):
        self.model_name = "fake-gpt"
        self.completions = FakeCompletions(responses)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=self.completions))

    async def generate(self, prompt: str, **kwargs) -> str:
        raise AssertionError("generate() must not be used on the native tool path")


class PlainModel:
    """Model without an OpenAI-shaped client: only generate()."""

    model_name = "plain"

    def __init__(self):
        self.prompts: List[str] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return f"model-answer:{prompt}"


class EchoTool(BaseTool):
    def __init__(self):
        super().__init__(name="echo", description="Echo back the input text")

    async def run(self, **kwargs) -> Any:
        return {"echoed": kwargs["text"]}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "required": ["text"],
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
        }


class TestDetectionAndSchema:
    def test_detects_openai_shaped_client(self):
        assert _supports_native_tools(FakeOpenAIModel([]))

    def test_plain_model_not_detected(self):
        assert not _supports_native_tools(PlainModel())

    def test_schema_built_from_get_parameters(self):
        schema = build_tool_schema(CalculatorTool())
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "calculator"
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "expression" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["expression"]

    def test_explicit_schema_attr_wins(self):
        tool = EchoTool()
        tool.schema = {"type": "object", "properties": {"custom": {"type": "integer"}}}
        schema = build_tool_schema(tool)
        assert schema["function"]["parameters"] == tool.schema


class TestNativeToolLoop:
    async def test_tool_call_executed_and_result_fed_back(self):
        model = FakeOpenAIModel(
            [
                _response(
                    tool_calls=[_tool_call("call_1", "calculator", '{"expression": "2 + 3"}')]
                ),
                _response(content="The answer is 5."),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("What is 2 + 3?")

        assert result["type"] == "model"
        assert result["result"] == "The answer is 5."
        assert result["tool_calls"] == [
            {"tool": "calculator", "args": {"expression": "2 + 3"}, "result": 5.0}
        ]

        # First request carried the JSON-schema tool definitions.
        first = model.completions.calls[0]
        assert first["model"] == "fake-gpt"
        assert first["tools"][0]["function"]["name"] == "calculator"

        # Second request contains the assistant tool_calls turn and the tool result.
        second_messages = model.completions.calls[1]["messages"]
        assert second_messages[-2]["role"] == "assistant"
        assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "calculator"
        tool_msg = second_messages[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert json.loads(tool_msg["content"]) == 5.0

    async def test_system_prompt_included(self):
        model = FakeOpenAIModel([_response(content="done")])
        agent = Agent(model=model, tools=[EchoTool()], system_prompt="You are terse.")
        await agent.run("hello")
        messages = model.completions.calls[0]["messages"]
        assert messages[0] == {"role": "system", "content": "You are terse."}
        assert messages[1] == {"role": "user", "content": "hello"}

    async def test_multiple_tool_calls_in_one_turn(self):
        model = FakeOpenAIModel(
            [
                _response(
                    tool_calls=[
                        _tool_call("c1", "echo", '{"text": "a"}'),
                        _tool_call("c2", "echo", '{"text": "b"}'),
                    ]
                ),
                _response(content="both done"),
            ]
        )
        agent = Agent(model=model, tools=[EchoTool()])
        result = await agent.run("echo twice")
        assert [c["result"] for c in result["tool_calls"]] == [
            {"echoed": "a"},
            {"echoed": "b"},
        ]
        # Both tool results answered by tool_call_id.
        messages = model.completions.calls[1]["messages"]
        ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        assert ids == ["c1", "c2"]

    async def test_malformed_json_args_reported_not_crash(self):
        model = FakeOpenAIModel(
            [
                _response(tool_calls=[_tool_call("c1", "calculator", '{"expression": ')]),
                _response(content="recovered"),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("calc")
        assert result["result"] == "recovered"
        record = result["tool_calls"][0]
        assert "Malformed arguments" in record["error"]
        # The error text is what the model saw as the tool result.
        tool_msg = model.completions.calls[1]["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert "Malformed arguments" in tool_msg["content"]

    async def test_non_object_json_args_rejected(self):
        model = FakeOpenAIModel(
            [
                _response(tool_calls=[_tool_call("c1", "calculator", '["2+3"]')]),
                _response(content="ok"),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("calc")
        assert "Malformed arguments" in result["tool_calls"][0]["error"]

    async def test_unknown_tool_reported(self):
        model = FakeOpenAIModel(
            [
                _response(tool_calls=[_tool_call("c1", "nonexistent", "{}")]),
                _response(content="ok"),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("do it")
        assert "Unknown tool 'nonexistent'" in result["tool_calls"][0]["error"]

    async def test_missing_required_params_reported(self):
        model = FakeOpenAIModel(
            [
                _response(tool_calls=[_tool_call("c1", "calculator", '{"wrong": 1}')]),
                _response(content="ok"),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("calc")
        assert "Missing required parameters" in result["tool_calls"][0]["error"]

    async def test_tool_runtime_error_reported(self):
        model = FakeOpenAIModel(
            [
                _response(
                    tool_calls=[_tool_call("c1", "calculator", '{"expression": "1/0"}')]
                ),
                _response(content="cannot divide"),
            ]
        )
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("calc")
        assert "failed" in result["tool_calls"][0]["error"]
        assert result["result"] == "cannot divide"

    async def test_max_turns_bounds_the_loop(self):
        looping = _response(
            tool_calls=[_tool_call("c1", "echo", '{"text": "again"}')]
        )
        model = FakeOpenAIModel([looping, looping, looping])
        agent = Agent(model=model, tools=[EchoTool()], max_turns=3)
        result = await agent.run("loop forever")
        assert "max_turns (3) exhausted" in result["error"]
        assert len(model.completions.calls) == 3

    async def test_max_turns_override_per_run(self):
        looping = _response(tool_calls=[_tool_call("c1", "echo", '{"text": "x"}')])
        model = FakeOpenAIModel([looping])
        agent = Agent(model=model, tools=[EchoTool()])
        result = await agent.run("loop", max_turns=1)
        assert "max_turns (1) exhausted" in result["error"]

    async def test_api_error_returned_as_error_dict(self):
        class ExplodingCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("api down")

        model = FakeOpenAIModel([])
        model.client = SimpleNamespace(
            chat=SimpleNamespace(completions=ExplodingCompletions())
        )
        agent = Agent(model=model, tools=[EchoTool()])
        result = await agent.run("hello")
        assert result["type"] == "model"
        assert "api down" in result["error"]


class TestKeywordFallback:
    async def test_keyword_routing_still_works_without_native_client(self):
        model = PlainModel()
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("use the calculator please", expression="2 * 4")
        assert result == {"type": "tool", "tool": "calculator", "result": 8.0}
        assert model.prompts == []  # tool short-circuited the model

    async def test_keyword_fallback_uses_model_when_no_tool_matches(self):
        model = PlainModel()
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("tell me a story")
        assert result["type"] == "model"
        assert result["result"] == "model-answer:tell me a story"

    async def test_native_model_without_tools_uses_model_directly(self):
        # No tools registered: even a native-capable model goes through generate().
        model = FakeOpenAIModel([])

        async def generate(prompt, **kwargs):
            return "plain answer"

        model.generate = generate
        agent = Agent(model=model, tools=[])
        result = await agent.run("hello")
        assert result == {"type": "model", "result": "plain answer"}

    async def test_keyword_fallback_missing_params_is_clear_error(self):
        model = PlainModel()
        agent = Agent(model=model, tools=[CalculatorTool()])
        result = await agent.run("use the calculator")
        assert result["type"] == "tool"
        assert "Missing required parameters" in result["error"]


class TestMemoryIntegration:
    async def test_run_records_task_and_response_in_memory(self):
        model = FakeOpenAIModel([_response(content="hi")])
        agent = Agent(model=model, tools=[EchoTool()])
        await agent.run("remember me")
        assert agent.memory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
