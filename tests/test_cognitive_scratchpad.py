"""Tests for CognitiveScratchpadMemory's step dependencies and revision history.

The LLM is mocked — no live API calls.
"""

from __future__ import annotations

import json

import pytest

from multimind.memory.cognitive_scratchpad import CognitiveScratchpadMemory


class ScriptedLLM:
    def __init__(self, response):
        self.response = response

    async def generate(self, prompt, **kwargs):
        return self.response


def steps_response(steps):
    return json.dumps(
        {
            "steps": [s for s, _, _ in steps],
            "step_types": [t for _, t, _ in steps],
            "confidence": [c for _, _, c in steps],
        }
    )


THREE_STEPS = steps_response(
    [
        ("identify the question", "analysis", 0.9),
        ("gather relevant facts", "retrieval", 0.8),
        ("compose the answer", "synthesis", 0.85),
    ]
)


@pytest.mark.asyncio
async def test_add_message_creates_chained_step_dependencies():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))

    await memory.add_message({"role": "user", "content": "What is 2+2?"})

    steps = memory.reasoning_steps
    assert len(steps) == 3
    assert steps[0]["dependencies"] == []
    assert steps[1]["dependencies"] == [steps[0]["id"]]
    assert steps[2]["dependencies"] == [steps[1]["id"]]


@pytest.mark.asyncio
async def test_get_dependency_chain_returns_ancestors_in_order():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "question"})
    steps = memory.reasoning_steps

    chain = await memory.get_dependency_chain(steps[2]["id"])

    assert chain == [steps[0]["id"], steps[1]["id"]]


@pytest.mark.asyncio
async def test_add_step_dependency_cross_chain():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "first"})
    await memory.add_message({"role": "user", "content": "second"})

    first_chain_last = memory.reasoning_steps[2]["id"]
    second_chain_first = memory.reasoning_steps[3]["id"]

    await memory.add_step_dependency(second_chain_first, first_chain_last)

    assert first_chain_last in memory.reasoning_steps[3]["dependencies"]


@pytest.mark.asyncio
async def test_add_step_dependency_rejects_self_dependency():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[0]["id"]

    with pytest.raises(ValueError, match="itself"):
        await memory.add_step_dependency(step_id, step_id)


@pytest.mark.asyncio
async def test_add_step_dependency_rejects_cycle():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    steps = memory.reasoning_steps
    # steps[2] already depends on steps[1] which depends on steps[0].
    # Making steps[0] depend on steps[2] would create a cycle.
    with pytest.raises(ValueError, match="cycle"):
        await memory.add_step_dependency(steps[0]["id"], steps[2]["id"])


@pytest.mark.asyncio
async def test_add_step_dependency_unknown_step_raises():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[0]["id"]

    with pytest.raises(ValueError, match="Unknown step"):
        await memory.add_step_dependency(step_id, "step_999")
    with pytest.raises(ValueError, match="Unknown step"):
        await memory.add_step_dependency("step_999", step_id)


@pytest.mark.asyncio
async def test_revise_step_archives_previous_content():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[0]["id"]
    original_content = memory.reasoning_steps[0]["content"]

    await memory.revise_step(step_id, "revised content", reason="clarified wording")

    assert memory.reasoning_steps[0]["content"] == "revised content"
    history = await memory.get_revision_history(step_id)
    assert len(history) == 1
    assert history[0]["content"] == original_content
    assert history[0]["reason"] == "clarified wording"


@pytest.mark.asyncio
async def test_revise_step_multiple_times_keeps_full_history():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[0]["id"]
    original_content = memory.reasoning_steps[0]["content"]

    await memory.revise_step(step_id, "v2")
    await memory.revise_step(step_id, "v3")

    history = await memory.get_revision_history(step_id)
    assert [h["content"] for h in history] == [original_content, "v2"]
    assert memory.reasoning_steps[0]["content"] == "v3"


@pytest.mark.asyncio
async def test_revise_step_unknown_raises():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    with pytest.raises(ValueError, match="Unknown step"):
        await memory.revise_step("nope", "content")


@pytest.mark.asyncio
async def test_get_revision_history_empty_for_unrevised_step():
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS))
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[0]["id"]

    history = await memory.get_revision_history(step_id)

    assert history == []


@pytest.mark.asyncio
async def test_save_and_load_persist_dependencies_and_history(tmp_path):
    storage_path = str(tmp_path / "scratchpad.json")
    memory = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS), storage_path=storage_path)
    await memory.add_message({"role": "user", "content": "q"})
    step_id = memory.reasoning_steps[1]["id"]
    await memory.revise_step(step_id, "revised")

    reloaded = CognitiveScratchpadMemory(llm=ScriptedLLM(THREE_STEPS), storage_path=storage_path)
    await reloaded.load()

    reloaded_step = next(s for s in reloaded.reasoning_steps if s["id"] == step_id)
    assert reloaded_step["content"] == "revised"
    assert len(reloaded_step["revision_history"]) == 1
    assert reloaded_step["dependencies"] == [reloaded.reasoning_steps[0]["id"]]
