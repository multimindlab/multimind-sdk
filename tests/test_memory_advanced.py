from datetime import timedelta

import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs

from multimind.memory.active_learning import ActiveLearningMemory
from multimind.memory.consensus import ConsensusMemory
from multimind.memory.generative import GenerativeMemory
from multimind.memory.implicit import ImplicitMemory
from multimind.memory.procedural import ProceduralMemory
from multimind.memory.reinforcement import ReinforcementMemory
from multimind.memory.semantic import SemanticMemory


class DummyLLM:
    async def generate(self, prompt, **kwargs):
        raise RuntimeError("LLM should not be called in these tests")


# --- Implicit memory ---


@pytest.mark.asyncio
async def test_implicit_memory_basic():
    mem = ImplicitMemory(skill_decay=0.5)
    await mem.add_skill("s1", "base skill", "the base skill", "coding")
    await mem.add_skill("s2", "advanced skill", "builds on base", "coding", prerequisites=["s1"])

    skill = await mem.get_skill("s2")
    assert skill["proficiency"] == 0.0

    await mem.record_practice("s2", performance_score=0.8, duration=10.0)
    skill = await mem.get_skill("s2")
    # (0.8 - 0.0) * (1 - 0.5) = 0.4
    assert skill["proficiency"] == pytest.approx(0.4)
    assert skill["practice_count"] == 1

    prereqs = await mem.get_prerequisites("s2")
    assert [p["id"] for p in prereqs] == ["s1"]

    stats = await mem.get_stats()
    assert stats["total_skills"] == 2
    assert stats["skill_graph_edges"] == 1


@pytest.mark.asyncio
async def test_implicit_memory_edge_cases(tmp_path):
    mem = ImplicitMemory(storage_path=str(tmp_path / "skills.json"))
    assert await mem.get_skill("missing") is None
    assert await mem.get_prerequisites("missing") == []
    assert await mem.get_skill_progress("missing") == {}

    await mem.add_skill("s1", "skill", "desc", "cat")
    await mem.remove_skill("s1")
    assert await mem.get_skill("s1") is None
    assert (await mem.get_stats())["total_skills"] == 0

    await mem.add_skill("s2", "skill2", "desc2", "cat")
    await mem.record_practice("s2", 0.9, 5.0)
    await mem.save()

    restored = ImplicitMemory(storage_path=str(tmp_path / "skills.json"))
    await restored.load()
    skill = await restored.get_skill("s2")
    assert skill["name"] == "skill2"
    assert skill["practice_count"] == 1


# --- Active learning memory ---


@pytest.mark.asyncio
async def test_active_learning_feedback_loop():
    mem = ActiveLearningMemory(
        DummyLLM(), enable_reinforcement=False, enable_feedback_analysis=False
    )
    await mem.add_message({"content": "python tips"})
    await mem.add_message({"content": "java tips"})

    await mem.record_feedback("python", ["item_0"], useful=True)
    await mem.record_feedback("python", ["item_0"], useful=True)
    await mem.record_feedback("python", ["item_1"], useful=False)

    items = {item["id"]: item for item in mem.items}
    # 0.5 -> 0.6 -> 0.68 for useful, 0.5 -> 0.4 for not useful
    assert items["item_0"]["metadata"]["utility"] == pytest.approx(0.68)
    assert items["item_1"]["metadata"]["utility"] == pytest.approx(0.4)
    assert items["item_0"]["metadata"]["feedback_count"] == 2

    results = await mem.retrieve("python tips", k=2)
    assert [r["id"] for r in results] == ["item_0", "item_1"]
    assert results[0]["retrieval_score"] > results[1]["retrieval_score"]


@pytest.mark.asyncio
async def test_active_learning_low_utility_eviction():
    mem = ActiveLearningMemory(
        DummyLLM(),
        enable_reinforcement=False,
        enable_feedback_analysis=False,
        learning_rate=0.9,
        eviction_utility=0.2,
    )
    await mem.add_message({"content": "junk memory"})
    await mem.add_message({"content": "good memory"})

    # 0.5 -> 0.05, below eviction threshold
    await mem.record_feedback("q", ["item_0"], useful=False)

    remaining = [item["id"] for item in mem.items]
    assert "item_0" not in remaining
    assert "item_1" in remaining
    assert all(f["item_id"] != "item_0" for f in mem.feedback)


# --- Consensus memory ---


@pytest.mark.asyncio
async def test_consensus_local_replication_and_majority_vote():
    a = ConsensusMemory("a")
    b = ConsensusMemory("b")
    c = ConsensusMemory("c")
    a.attach_replicas([b, c])

    await a.add_memory("m1", "hello")
    assert (await b.get_memory("m1"))["content"] == "hello"
    assert (await c.get_memory("m1"))["content"] == "hello"

    result = await a.get_majority_memory("m1")
    assert result["content"] == "hello"
    assert result["votes"] == 3

    # A diverging replica is outvoted
    b.memories["m1"]["content"] = "tampered"
    result = await a.get_majority_memory("m1")
    assert result["content"] == "hello"
    assert result["votes"] == 2

    await a.update_memory("m1", {"content": "updated"})
    assert (await c.get_memory("m1"))["content"] == "updated"

    await a.remove_memory("m1")
    assert await c.get_memory("m1") is None
    assert await a.get_majority_memory("m1") is None


@pytest.mark.asyncio
async def test_consensus_network_raft_fails_honest():
    follower = ConsensusMemory("f", nodes=["f", "leader"])
    with pytest.raises(NotImplementedError):
        await follower.add_memory("m1", "content")
    with pytest.raises(NotImplementedError):
        await follower.start_background_tasks()
    with pytest.raises(NotImplementedError):
        await follower._request_vote("leader")
    with pytest.raises(NotImplementedError):
        await follower._forward_to_leader("ADD_MEMORY", {})


@pytest.mark.asyncio
async def test_consensus_base_memory_interface():
    node = ConsensusMemory("solo")
    await node.add_message({"role": "user", "content": "hi"})
    messages = await node.get_messages()
    assert messages == [{"role": "user", "content": "hi"}]
    await node.clear()
    assert await node.get_messages() == []


# --- Reinforcement memory ---


@pytest.mark.asyncio
async def test_reinforcement_q_learning_frees_budget():
    mem = ReinforcementMemory(total_budget=100, min_budget=30, max_budget=100, epsilon=0.0)
    await mem.add_memory("a", "x" * 50)

    # Q-table starts at zero: greedy tie-break picks KEEP, which frees nothing
    # and earns a negative reward under budget pressure.
    with pytest.raises(MemoryError):
        await mem.add_memory("b", "y" * 60)
    assert mem.optimization_rounds == 1

    # After the Q-update, REMOVE beats the penalized KEEP and frees the budget.
    await mem.add_memory("b", "y" * 60)
    assert "a" not in mem.memories
    assert "b" in mem.memories

    stats = await mem.get_stats()
    assert stats["optimization_rounds"] == 2
    assert stats["q_table_size"] >= 1


@pytest.mark.asyncio
async def test_reinforcement_access_rewards_and_interface(tmp_path):
    mem = ReinforcementMemory(storage_path=str(tmp_path / "rl.json"))
    await mem.add_message({"role": "user", "content": "remember this"})
    messages = await mem.get_messages()
    assert messages == [{"role": "user", "content": "remember this"}]

    memory = await mem.get_memory("message_0")
    assert memory["access_count"] == 1
    assert mem.reward_history and mem.reward_history[0] > 0

    await mem.save()
    restored = ReinforcementMemory(storage_path=str(tmp_path / "rl.json"))
    await restored.load()
    assert (await restored.get_memory("message_0"))["content"] == "remember this"


# --- Generative memory ---


@pytest.mark.asyncio
async def test_generative_memory_regeneration_tracking():
    mem = GenerativeMemory()
    await mem.add_memory("g1", "original content", category="facts")

    await mem.regenerate_memory("g1", "regenerated content", confidence=0.9)
    memory = await mem.get_memory("g1")
    assert memory["content"] == "regenerated content"
    assert memory["original_content"] == "original content"
    assert memory["regeneration_count"] == 1

    history = await mem.get_regeneration_history("g1")
    assert len(history) == 1
    assert history[0]["old_content"] == "original content"

    stats = await mem.get_memory_stats("g1")
    assert stats["reconstruction_score"] == 0.9
    assert stats["avg_confidence"] == pytest.approx(0.9)
    assert 0.0 < stats["content_drift"] < 1.0


@pytest.mark.asyncio
async def test_generative_memory_due_regeneration():
    mem = GenerativeMemory(regeneration_interval=timedelta(seconds=0))
    await mem.add_memory("g1", "stale content", category="facts")
    assert await mem.check_regeneration_needed("g1") is True
    assert await mem.get_memories_needing_regeneration() == ["g1"]

    async def regenerate(memory):
        return memory["content"] + " (refreshed)", 0.8

    regenerated = await mem.regenerate_due_memories(regenerate)
    assert regenerated == ["g1"]
    assert (await mem.get_memory("g1"))["content"] == "stale content (refreshed)"
    assert mem.reconstruction_scores["g1"] == 0.8


@pytest.mark.asyncio
async def test_generative_memory_interface():
    mem = GenerativeMemory()
    await mem.add_message({"role": "user", "content": "hello"})
    assert await mem.get_messages() == [{"role": "user", "content": "hello"}]
    stats = await mem.get_stats()
    assert stats["total_memories"] == 1
    await mem.clear()
    assert (await mem.get_stats())["total_memories"] == 0


# --- Pre-existing component memory smoke tests ---


def test_procedural_memory_basic():
    class DummyLLM:
        pass

    mem = ProceduralMemory(DummyLLM(), max_procedures=5)
    assert mem is not None


def test_semantic_memory_basic():
    class DummyLLM:
        pass

    mem = SemanticMemory(DummyLLM(), max_concepts=3)
    assert mem is not None
