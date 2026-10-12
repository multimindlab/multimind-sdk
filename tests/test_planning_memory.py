"""Tests for PlanningMemory's honest core: plan-step recording and
success-weighted next-step suggestion. No mocked LLM needed — pure stats."""

from __future__ import annotations

import pytest

from multimind.memory.planning import PlanningMemory


@pytest.mark.asyncio
async def test_add_memory_and_get_memory_roundtrip():
    memory = PlanningMemory()

    await memory.add_memory("m1", "did something", state={"loc": "a"}, action="move")
    fetched = await memory.get_memory("m1")

    assert fetched["content"] == "did something"
    assert fetched["access_count"] == 1


@pytest.mark.asyncio
async def test_add_memory_with_full_context_registers_plan():
    memory = PlanningMemory()

    await memory.add_memory(
        "m1", "moved", state={"loc": "a"}, action="move", outcome={"success": True}
    )

    assert "m1" in memory.plans
    assert memory.plans["m1"]["success"] is True


@pytest.mark.asyncio
async def test_suggest_next_action_prefers_higher_success_rate():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})
    await memory.add_memory(
        "m2", "x", state={"loc": "a"}, action="move", outcome={"success": False}
    )
    await memory.add_memory("m3", "x", state={"loc": "a"}, action="wait", outcome={"success": True})

    suggestion = await memory.suggest_next_action({"loc": "a"})

    assert suggestion["action"] == "wait"
    assert suggestion["success_rate"] == 1.0
    assert suggestion["support"] == 1


@pytest.mark.asyncio
async def test_suggest_next_action_no_similar_states_returns_none():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})

    suggestion = await memory.suggest_next_action({"loc": "totally different"})

    assert suggestion is None


@pytest.mark.asyncio
async def test_suggest_next_action_ignores_dissimilar_states_below_threshold():
    memory = PlanningMemory(similarity_threshold=0.9)
    await memory.add_memory(
        "m1", "x", state={"loc": "a", "extra": 1}, action="move", outcome={"success": True}
    )

    # Only one of two keys matches -> similarity 0.5, below the 0.9 threshold.
    suggestion = await memory.suggest_next_action({"loc": "a", "extra": 999})

    assert suggestion is None


@pytest.mark.asyncio
async def test_record_plan_outcome_updates_plan_and_stats():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})

    await memory.record_plan_outcome("m1", success=False, actual_outcome={"success": False})

    assert memory.plans["m1"]["success"] is False
    stats = await memory.get_plan_stats("m1")
    assert stats["success_rate"] == 0.0
    assert stats["total_executions"] == 1


@pytest.mark.asyncio
async def test_get_similar_plans_filters_by_min_similarity():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})
    await memory.add_memory("m2", "y", state={"loc": "b"}, action="move", outcome={"success": True})

    plans = await memory.get_similar_plans({"loc": "a"}, min_similarity=1.0)

    assert len(plans) == 1
    assert plans[0]["state"] == {"loc": "a"}


@pytest.mark.asyncio
async def test_get_stats_aggregates_plan_success():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})
    await memory.record_plan_outcome("m1", success=True, actual_outcome={"success": True})
    await memory.record_plan_outcome("m1", success=False, actual_outcome={"success": False})

    stats = await memory.get_stats()

    assert stats["total_memories"] == 1
    assert stats["total_plans"] == 1
    assert stats["avg_success_rate"] == 0.5


@pytest.mark.asyncio
async def test_base_memory_interface_add_message_and_get_messages():
    memory = PlanningMemory()

    await memory.add_message({"role": "user", "content": "hello"})
    await memory.add_message({"role": "assistant", "content": "hi there"})
    messages = await memory.get_messages()

    assert [m["content"] for m in messages] == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_clear_resets_all_state():
    memory = PlanningMemory()
    await memory.add_memory("m1", "x", state={"loc": "a"}, action="move", outcome={"success": True})

    await memory.clear()

    assert memory.memories == {}
    assert memory.plans == {}
    assert await memory.get_stats() == {
        "total_memories": 0,
        "total_plans": 0,
        "avg_success_rate": 0.0,
    }


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(tmp_path):
    storage_path = str(tmp_path / "planning.json")
    memory = PlanningMemory(storage_path=storage_path)
    await memory.add_memory(
        "m1", "did something", state={"loc": "a"}, action="move", outcome={"success": True}
    )
    await memory.record_plan_outcome("m1", success=True, actual_outcome={"success": True})
    await memory.save()

    reloaded = PlanningMemory(storage_path=storage_path)
    await reloaded.load()

    assert reloaded.memories["m1"]["content"] == "did something"
    assert reloaded.plans["m1"]["success"] is True
    assert reloaded.plan_success["m1"] == [True]


def test_calculate_state_similarity_jaccard():
    assert PlanningMemory._calculate_state_similarity({}, {}) == 1.0
    assert PlanningMemory._calculate_state_similarity({"a": 1}, {"a": 1}) == 1.0
    assert PlanningMemory._calculate_state_similarity({"a": 1}, {"a": 2}) == 0.0
    assert PlanningMemory._calculate_state_similarity({"a": 1, "b": 2}, {"a": 1}) == 0.5
