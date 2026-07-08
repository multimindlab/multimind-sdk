"""Unit tests for DeclarativeMemory temporal/causal/graph analysis methods."""

from datetime import datetime, timedelta

import pytest
pytest.importorskip("numpy")  # requires optional extras absent on core-only installs

from multimind.memory.declarative import DeclarativeMemory


class FakeLLM:
    async def generate(self, prompt, **kwargs):
        return "{}"

    async def embeddings(self, text, **kwargs):
        return [0.0] * 4


def make_memory(**kwargs):
    return DeclarativeMemory(llm=FakeLLM(), **kwargs)


async def add_fact(memory, content, timestamp=None):
    await memory.add_message({"role": "user", "content": content})
    fact = memory.facts[-1]
    if timestamp is not None:
        fact["timestamp"] = timestamp.isoformat()
    return fact["id"]


@pytest.mark.asyncio
async def test_temporal_relations_before_after_concurrent():
    memory = make_memory()
    now = datetime.now()
    early = await add_fact(memory, "first", now - timedelta(hours=1))
    near = await add_fact(memory, "second", now - timedelta(seconds=10))
    latest = await add_fact(memory, "third", now)

    await memory._analyze_temporal_relations(latest)

    record = memory.temporal_relations[latest]
    assert record["type"] == "timestamp_ordering"
    assert record["score"] == 1.0
    by_fact = {r["fact_id"]: r["relation"] for r in record["relations"]}
    assert by_fact[early] == "temporally_after"  # latest happened after early
    assert by_fact[near] == "temporally_during"  # within concurrency window
    fact = next(f for f in memory.facts if f["id"] == latest)
    assert fact["metadata"]["temporal_data"] == record


@pytest.mark.asyncio
async def test_temporal_relations_no_peers_scores_zero():
    memory = make_memory()
    only = await add_fact(memory, "solo")
    await memory._analyze_temporal_relations(only)
    assert memory.temporal_relations[only]["score"] == 0.0
    assert memory.temporal_relations[only]["relations"] == []


@pytest.mark.asyncio
async def test_causal_chains_lexical_cues():
    memory = make_memory()
    now = datetime.now()
    cause = await add_fact(memory, "the server crashed", now - timedelta(minutes=5))
    effect = await add_fact(memory, "the outage happened because the server crashed", now)

    await memory._analyze_causal_chains(effect)

    record = memory.causal_chains[effect]
    assert record["method"] == "lexical_heuristic"
    assert record["score"] == 1.0
    assert len(record["chains"]) == 1
    chain = record["chains"][0]
    assert chain["chain_type"] == "lexical_cue"
    assert "because" in chain["cues"]
    assert cause in chain["candidate_causes"]


@pytest.mark.asyncio
async def test_causal_chains_no_cues():
    memory = make_memory()
    fact_id = await add_fact(memory, "the sky is blue")
    await memory._analyze_causal_chains(fact_id)
    record = memory.causal_chains[fact_id]
    assert record["score"] == 0.0
    assert record["chains"] == []


@pytest.mark.asyncio
async def test_knowledge_graph_dict_fallback():
    memory = make_memory()
    now = datetime.now()
    first = await add_fact(memory, "first", now - timedelta(hours=1))
    second = await add_fact(memory, "second because first", now)

    await memory._analyze_temporal_relations(second)
    await memory._analyze_causal_chains(second)
    await memory._update_knowledge_graph(second)

    node = memory.knowledge_graph[second]
    assert node["type"] == "fact"
    assert node["content"] == "second because first"
    assert first in node["edges"]["temporally_after"]
    assert first in node["edges"]["candidate_causes"]


@pytest.mark.asyncio
async def test_knowledge_graph_delegates_to_graph_memory():
    class FakeGraphMemory:
        def __init__(self):
            self.messages = []

        async def add_message(self, message):
            self.messages.append(message)

    graph = FakeGraphMemory()
    memory = make_memory(graph_memory=graph)
    fact_id = await add_fact(memory, "delegated fact")

    await memory._update_knowledge_graph(fact_id)

    assert graph.messages == [{"role": "declarative_memory", "content": "delegated fact"}]
    fact = next(f for f in memory.facts if f["id"] == fact_id)
    assert fact["metadata"]["graph_data"]["delegated_to"] == "FakeGraphMemory"
    assert memory.knowledge_graph == {}


@pytest.mark.asyncio
async def test_scheduled_analysis_paths_do_not_crash():
    # Force the interval checks to fire on every add_message.
    memory = make_memory(temporal_interval=-1, causal_interval=-1, graph_update_interval=-1)
    await memory.add_message({"role": "user", "content": "a happened"})
    await memory.add_message({"role": "user", "content": "b happened because a happened"})

    assert "fact_1" in memory.temporal_relations
    assert "fact_1" in memory.causal_chains
    assert "fact_1" in memory.knowledge_graph

    stats = await memory.get_declarative_memory_stats()
    assert stats["temporal_stats"]["total_relations"] == 2
    assert stats["causal_stats"]["total_chains"] == 1
    assert stats["graph_stats"]["total_nodes"] == 2
