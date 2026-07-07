"""Tests for GraphRetriever — scripted LLM extraction, no live API calls."""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "multimind.rag", exc_type=ImportError
)  # requires optional extras absent on core-only installs
from multimind.rag.graph_retrieval import GraphRetriever


class ScriptedModel:
    """Returns queued responses from generate(), in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


def extraction(entities, relationships=()):
    return json.dumps(
        {
            "entities": [
                {"id": f"e_{name.lower()}", "name": name, "type": "concept"} for name in entities
            ],
            "relationships": [
                {"source": f"e_{s.lower()}", "target": f"e_{t.lower()}", "type": rel}
                for s, t, rel in relationships
            ],
        }
    )


def query_entities(*names):
    return json.dumps(
        [{"id": f"q_{name.lower()}", "name": name, "type": "concept"} for name in names]
    )


DOCS = [
    {"id": "d1", "content": "Python was created by Guido van Rossum."},
    {"id": "d2", "content": "Guido van Rossum worked at Dropbox."},
    {"id": "d3", "content": "Rust is a systems language."},
]

# One extraction response per document, in order.
DOC_EXTRACTIONS = [
    extraction(["Python", "Guido"], [("Guido", "Python", "created")]),
    extraction(["Guido", "Dropbox"], [("Guido", "Dropbox", "worked_at")]),
    extraction(["Rust"]),
]


@pytest.mark.asyncio
async def test_entity_overlap_ranking():
    model = ScriptedModel(DOC_EXTRACTIONS + [query_entities("Python", "Guido")])
    retriever = GraphRetriever(model, documents=DOCS)

    results = await retriever.retrieve("Who created Python, Guido?", k=5)

    # d1 contains both query entities (2.0); d2 contains Guido directly (1.0)
    # plus neighbors of Python/Guido (Dropbox -> d2 at 0.5, twice: via Python's
    # neighbor Guido is matched, Dropbox neighbor of Guido).
    ids = [doc["id"] for doc in results]
    assert ids[0] == "d1"
    assert "d2" in ids
    assert "d3" not in ids
    assert results[0]["score"] > results[1]["score"]


@pytest.mark.asyncio
async def test_neighbor_expansion_pulls_related_document():
    model = ScriptedModel(DOC_EXTRACTIONS + [query_entities("Dropbox")])
    retriever = GraphRetriever(model, documents=DOCS)

    results = await retriever.retrieve("Tell me about Dropbox", k=5)

    ids = [doc["id"] for doc in results]
    # d2 contains Dropbox directly; d1 is pulled in via the neighbor Guido.
    assert ids[0] == "d2"
    assert "d1" in ids
    assert "d3" not in ids
    d1 = next(doc for doc in results if doc["id"] == "d1")
    d2 = next(doc for doc in results if doc["id"] == "d2")
    assert d2["score"] > d1["score"]


@pytest.mark.asyncio
async def test_deterministic_given_same_script():
    async def run():
        model = ScriptedModel(DOC_EXTRACTIONS + [query_entities("Guido")])
        retriever = GraphRetriever(model, documents=DOCS)
        return await retriever.retrieve("Guido", k=5)

    first, second = await run(), await run()
    assert [(d["id"], d["score"]) for d in first] == [(d["id"], d["score"]) for d in second]


@pytest.mark.asyncio
async def test_k_limits_results():
    model = ScriptedModel(DOC_EXTRACTIONS + [query_entities("Guido")])
    retriever = GraphRetriever(model, documents=DOCS)

    results = await retriever.retrieve("Guido", k=1)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_no_match_returns_empty():
    model = ScriptedModel(DOC_EXTRACTIONS + [query_entities("Haskell")])
    retriever = GraphRetriever(model, documents=DOCS)

    assert await retriever.retrieve("Haskell", k=5) == []


@pytest.mark.asyncio
async def test_strict_parse_on_document_extraction():
    model = ScriptedModel(["this is not json"])
    retriever = GraphRetriever(model, documents=[DOCS[0]])

    with pytest.raises(ValueError, match="knowledge extraction"):
        await retriever.build()


@pytest.mark.asyncio
async def test_strict_parse_on_query_extraction():
    model = ScriptedModel(DOC_EXTRACTIONS + ["nope"])
    retriever = GraphRetriever(model, documents=DOCS)

    with pytest.raises(ValueError, match="entities"):
        await retriever.retrieve("anything")


@pytest.mark.asyncio
async def test_string_documents_and_incremental_add():
    model = ScriptedModel(
        [
            extraction(["Coffee"]),
            extraction(["Tea"]),
            query_entities("Tea"),
        ]
    )
    retriever = GraphRetriever(model)
    await retriever.add_documents(["Coffee is a drink."])
    await retriever.add_documents(["Tea is also a drink."])

    results = await retriever.retrieve("Tea", k=5)

    assert [doc["id"] for doc in results] == ["doc_1"]
    assert results[0]["content"] == "Tea is also a drink."


def test_invalid_document_shape_rejected():
    with pytest.raises(ValueError, match="content"):
        GraphRetriever(ScriptedModel([]), documents=[{"id": "x"}])


@pytest.mark.asyncio
async def test_duplicate_document_id_rejected():
    model = ScriptedModel([extraction(["A"]), extraction(["A"])])
    retriever = GraphRetriever(model, documents=[{"id": "d", "content": "a"}])
    await retriever.build()

    with pytest.raises(ValueError, match="Duplicate"):
        await retriever.add_documents([{"id": "d", "content": "again"}])
