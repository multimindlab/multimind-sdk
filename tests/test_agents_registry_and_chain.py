"""Tests for AgentRegistry and the sequential ReasoningChain example."""

import runpy
from pathlib import Path

import pytest

from multimind.agents import AgentRegistry
from multimind.agents.react_toolchain import ReasoningChainExecutionError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agents" / "react_toolchain_example.py"


def _agent_ok(query, state=None):
    return {"response": f"ok: {query}", "state": {"last": query}}


def _agent_fail(query, state=None):
    raise ValueError("boom")


def test_registry_register_and_run_updates_state():
    registry = AgentRegistry()
    registry.register_agent("ok", _agent_ok)

    assert registry.get_agent("ok") is _agent_ok
    assert registry.get_agent("missing") is None
    assert registry.run_agent("missing", "q") is None

    result = registry.run_agent("ok", "hello", session_id="s1")
    assert result["response"] == "ok: hello"
    assert registry.get_state("s1") == {"last": "hello"}

    registry.set_state("s2", {"x": 1})
    assert registry.get_state("s2") == {"x": 1}


def test_registry_fallback_on_failure():
    registry = AgentRegistry()
    registry.register_agent("a", _agent_fail)
    registry.register_agent("b", _agent_ok)
    registry.set_fallback("a", "b")

    assert registry.run_agent("a", "q")["response"] == "ok: q"


def test_registry_fallback_cycle_is_aborted():
    registry = AgentRegistry()
    registry.register_agent("a", _agent_fail)
    registry.register_agent("b", _agent_fail)
    registry.set_fallback("a", "b")
    registry.set_fallback("b", "a")

    assert registry.run_agent("a", "q") is None


def test_reasoning_chain_example_runs_offline(capsys):
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    out = capsys.readouterr().out
    assert "Final result: The answer is 4" in out


def test_reasoning_chain_example_rejects_non_arithmetic():
    mod = runpy.run_path(str(EXAMPLE))
    with pytest.raises(ReasoningChainExecutionError):
        mod["chain"].run("What is the capital of France?")
