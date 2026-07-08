"""
Behavioral tests for the "Beta" memory modules: EpisodicMemory, SemanticMemory,
ProceduralMemory, HybridMemory, and VectorStoreMemory.

These tests exercise real add/retrieve/expiry/routing round-trips against the
actual module code, mocking only the LLM boundary (generate/embeddings) with
small deterministic fakes that match the exact "Key: value" prompt-response
formats each module's parser expects.
"""

import json

import pytest

# HybridMemory/KnowledgeGraphMemory/VectorStoreMemory need numpy/networkx
# (the [memory] extras), which test-core doesn't install.
pytest.importorskip("numpy", reason="requires multimind-sdk[memory]")
pytest.importorskip("networkx", reason="requires multimind-sdk[memory]")

from multimind.memory.episodic import EpisodicMemory  # noqa: E402
from multimind.memory.hybrid import HybridMemory  # noqa: E402
from multimind.memory.knowledge_graph import KnowledgeGraphMemory  # noqa: E402
from multimind.memory.procedural import ProceduralMemory  # noqa: E402
from multimind.memory.semantic import SemanticMemory  # noqa: E402
from multimind.memory.time_weighted import TimeWeightedMemory  # noqa: E402
from multimind.memory.vector_store import VectorStoreMemory  # noqa: E402
from multimind.vector_store.base import VectorStoreConfig, VectorStoreType  # noqa: E402

# ---------------------------------------------------------------------------
# Fake LLMs: deterministic, format-matching responses per memory type.
# ---------------------------------------------------------------------------


class EpisodicFakeLLM:
    """Fake LLM matching EpisodicMemory's prompt/response formats."""

    async def generate(self, prompt, *args, **kwargs):
        if "determine the intensity (0-1) of each emotion" in prompt:
            return "Emotion: joy\nIntensity: 0.8\n---\nEmotion: calm\nIntensity: 0.4\n---\n"
        if "office" in prompt:
            return (
                "Location: office\n"
                "Emotions: stress\n"
                "Participants: bob\n"
                "Confidence: 0.9\n"
                "Importance: 0.9\n"
                "Emotional Intensity: 0.6\n"
            )
        return (
            "Location: kitchen\n"
            "Emotions: joy, calm\n"
            "Participants: alice\n"
            "Confidence: 0.9\n"
            "Importance: 0.8\n"
            "Emotional Intensity: 0.7\n"
        )

    async def embeddings(self, text, **kwargs):
        # Identical vector for every episode -> similarity 1.0, guaranteeing
        # deterministic chaining/consolidation behavior.
        return [1.0, 0.0, 0.0]


class SemanticFakeLLM:
    """Fake LLM matching SemanticMemory's prompt/response formats."""

    async def generate(self, prompt, *args, **kwargs):
        if "Determine the relationship type between these concepts" in prompt:
            return "related_to"
        if "infer new relationships" in prompt:
            return "Related Concept: neural network\nRelationship Type: related_to\nConfidence: 0.85\n---\n"
        if "Validate this concept" in prompt:
            return "Valid: true\nConfidence: 0.9\nIssues: none\n"
        return (
            "Concept: machine learning\n"
            "Category: technology\n"
            "Properties: supervised, unsupervised\n"
            "Confidence: 0.9\n"
            "---\n"
            "Concept: neural network\n"
            "Category: technology\n"
            "Properties: layers, weights\n"
            "Confidence: 0.85\n"
            "---\n"
        )

    async def embeddings(self, text, **kwargs):
        # Identical vector -> concepts extracted together are always "related".
        return [1.0, 0.0]


class ProceduralFakeLLM:
    """Fake LLM matching ProceduralMemory's prompt/response formats."""

    async def generate(self, prompt, *args, **kwargs):
        if "Adapt this procedure" in prompt:
            return "Steps:\n1. Rebuild artifact\n2. Retry deploy\n"
        if "Optimize this procedure" in prompt:
            return "Steps:\n1. Optimized step 1\n2. Optimized step 2\n"
        if "Validate this procedure" in prompt:
            return "Valid: true\nConfidence: 0.9\nIssues: none\n"
        if "run the test suite" in prompt:
            return (
                "Procedure: Run tests\n"
                "Category: qa\n"
                "Prerequisites: pytest\n"
                "Expected Outcome: tests pass\n"
                "Steps:\n"
                "1. Install deps\n"
                "2. Run pytest\n"
                "Confidence: 0.85\n"
            )
        return (
            "Procedure: Deploy service\n"
            "Category: devops\n"
            "Prerequisites: git, docker\n"
            "Expected Outcome: service running\n"
            "Steps:\n"
            "1. Build image\n"
            "2. Push image\n"
            "3. Deploy to cluster\n"
            "Confidence: 0.9\n"
        )

    async def embeddings(self, text, **kwargs):
        # Identical vector -> procedures are always mutually "related" for chaining.
        return [1.0, 0.0]


class HybridFakeLLM:
    """Fake LLM matching HybridMemory's JSON prompt/response formats."""

    def __init__(self, route_to):
        self.route_to = route_to

    async def generate(self, prompt, *args, **kwargs):
        if "Route message to appropriate memory types" in prompt:
            return json.dumps(
                {
                    "selected_memories": self.route_to,
                    "routing_reason": "matched by fake router",
                    "confidence": 0.9,
                }
            )
        if "Analyze routing performance" in prompt:
            return json.dumps({"learning_updates": {}, "learning_reason": "no updates"})
        # Any other (analysis/optimization/etc.) prompt is not expected to
        # fire within these short-lived tests since their intervals are 3600s.
        return "{}"

    async def embeddings(self, text, **kwargs):
        return [0.0]


class VectorFakeLLM:
    """Fake LLM returning small orthogonal unit vectors per distinct text."""

    def __init__(self):
        self._vectors = {
            "apple pie recipe": [1.0, 0.0, 0.0, 0.0],
            "banana bread recipe": [0.0, 1.0, 0.0, 0.0],
            "car engine repair": [0.0, 0.0, 1.0, 0.0],
        }

    async def generate(self, prompt, *args, **kwargs):
        return ""

    async def embeddings(self, text, **kwargs):
        if text in self._vectors:
            return self._vectors[text]
        if text == "apple-like fruit dessert":
            return [0.9, 0.1, 0.0, 0.0]
        # Unknown text: distinct orthogonal-ish vector in the 4th dimension.
        return [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_add_retrieve_indices_and_clear(self, tmp_path):
        mem = EpisodicMemory(
            llm=EpisodicFakeLLM(),
            storage_path=str(tmp_path / "episodic.json"),
        )

        await mem.add_message({"content": "Cooking dinner at home"})
        await mem.add_message({"content": "Stressful meeting at the office"})

        messages = await mem.get_messages()
        assert len(messages) == 2
        contents = {m["content"] for m in messages}
        assert contents == {"Cooking dinner at home", "Stressful meeting at the office"}

        # Spatial index
        kitchen_eps = await mem.get_episodes_by_location("kitchen")
        office_eps = await mem.get_episodes_by_location("office")
        assert len(kitchen_eps) == 1
        assert kitchen_eps[0]["content"] == "Cooking dinner at home"
        assert len(office_eps) == 1
        assert office_eps[0]["content"] == "Stressful meeting at the office"

        # Emotional index
        joy_eps = await mem.get_episodes_by_emotion("joy")
        stress_eps = await mem.get_episodes_by_emotion("stress")
        assert len(joy_eps) == 1
        assert len(stress_eps) == 1

        # Temporal index (both created "today")
        today = messages[0]["timestamp"].split("T")[0]
        date_eps = await mem.get_episodes_by_date(today)
        assert len(date_eps) == 2

        # get_episode_by_id
        ep0 = await mem.get_episode_by_id("ep_0")
        assert ep0 is not None
        assert ep0["content"] == "Cooking dinner at home"
        assert await mem.get_episode_by_id("nonexistent") is None

        # Emotional profile (per-episode intensity breakdown)
        profile = await mem.get_emotional_profile("ep_0")
        assert profile == {"joy": 0.8, "calm": 0.4}
        filtered = await mem.get_emotional_profile("ep_0", min_intensity=0.5)
        assert filtered == {"joy": 0.8}

        # Chaining: identical embeddings mean ep_1's most-related predecessor is
        # ep_0; ep_1's chain accumulates ep_0's (empty) lineage plus itself.
        chain = await mem.get_episode_chain("ep_1")
        assert len(chain) == 1
        assert chain[0]["id"] == "ep_1"
        assert await mem.get_episode_chain("ep_0") == []

        # Stats + suggestions must not raise (regression check for the
        # len()-on-int bug in get_episode_suggestions).
        stats = await mem.get_episode_stats()
        assert stats["total_episodes"] == 2
        suggestions = await mem.get_episode_suggestions()
        assert isinstance(suggestions, list)

        await mem.clear()
        assert await mem.get_messages() == []
        assert await mem.get_episodes_by_location("kitchen") == []

    @pytest.mark.asyncio
    async def test_save_load_round_trip(self, tmp_path):
        storage_path = str(tmp_path / "episodic.json")
        mem1 = EpisodicMemory(llm=EpisodicFakeLLM(), storage_path=storage_path)
        await mem1.add_message({"content": "Cooking dinner at home"})

        mem2 = EpisodicMemory(llm=EpisodicFakeLLM(), storage_path=storage_path)
        await mem2.load()

        messages = await mem2.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "Cooking dinner at home"
        assert len(mem2.episode_embeddings) == 1
        kitchen_eps = await mem2.get_episodes_by_location("kitchen")
        assert len(kitchen_eps) == 1


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_add_retrieve_relationships_and_clear(self, tmp_path):
        mem = SemanticMemory(
            llm=SemanticFakeLLM(),
            storage_path=str(tmp_path / "semantic.json"),
        )

        await mem.add_message({"content": "Machine learning uses neural networks"})

        messages = await mem.get_messages()
        assert len(messages) == 2
        contents = {m["content"] for m in messages}
        assert contents == {"machine learning", "neural network"}

        concept0 = await mem.get_concept_by_id("concept_0")
        assert concept0["metadata"]["category"] == "technology"
        assert concept0["metadata"]["properties"] == {"supervised", "unsupervised"}
        assert await mem.get_concept_by_id("nonexistent") is None

        # Relationship inference: identical embeddings force concept_0 and
        # concept_1 to be related to each other.
        related = await mem.get_related_concepts("concept_0")
        assert len(related) == 1
        assert related[0]["concept"]["id"] == "concept_1"
        assert related[0]["relationship_type"] == "related_to"

        # Inference cache populated for both concepts.
        inferred = await mem.get_inferred_relationships("concept_0")
        assert len(inferred) == 1
        assert inferred[0]["concept"] == "neural network"
        filtered = await mem.get_inferred_relationships("concept_0", min_confidence=0.9)
        assert filtered == []

        stats = await mem.get_concept_stats()
        assert stats["total_concepts"] == 2
        assert stats["category_distribution"]["technology"] == 2

        suggestions = await mem.get_concept_suggestions()
        assert isinstance(suggestions, list)

        await mem.clear()
        assert await mem.get_messages() == []
        assert await mem.get_concept_by_id("concept_0") is None

    @pytest.mark.asyncio
    async def test_save_load_round_trip(self, tmp_path):
        storage_path = str(tmp_path / "semantic.json")
        mem1 = SemanticMemory(llm=SemanticFakeLLM(), storage_path=storage_path)
        await mem1.add_message({"content": "Machine learning uses neural networks"})

        mem2 = SemanticMemory(llm=SemanticFakeLLM(), storage_path=storage_path)
        await mem2.load()

        messages = await mem2.get_messages()
        assert len(messages) == 2
        # Regression check: load() must actually await embeddings() so this
        # list holds real vectors, not un-awaited coroutine objects.
        assert len(mem2.concept_embeddings) == 2
        assert all(isinstance(v, list) for v in mem2.concept_embeddings)
        related = await mem2.get_related_concepts("concept_0")
        assert len(related) == 1


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------


class TestProceduralMemory:
    @pytest.mark.asyncio
    async def test_add_execute_adapt_and_clear(self, tmp_path):
        mem = ProceduralMemory(
            llm=ProceduralFakeLLM(),
            storage_path=str(tmp_path / "procedural.json"),
        )

        await mem.add_message({"content": "How to deploy the service to production"})
        await mem.add_message({"content": "How to run the test suite before merging"})

        messages = await mem.get_messages()
        assert len(messages) == 2
        contents = {m["content"] for m in messages}
        assert contents == {"Deploy service", "Run tests"}

        proc0 = await mem.get_procedure_by_id("proc_0")
        assert proc0["steps"] == ["Build image", "Push image", "Deploy to cluster"]
        assert await mem.get_procedure_by_id("nonexistent") is None

        # Chaining: identical embeddings force proc_1 to link back to proc_0.
        chain = await mem.get_procedure_chain("proc_1")
        assert len(chain) == 1
        assert chain[0]["id"] == "proc_0"

        # Record a failed execution -> triggers adaptation (steps blended).
        await mem.record_execution("proc_0", success=False, duration=12.0, notes="timeout")
        history = await mem.get_execution_history("proc_0")
        assert len(history) == 1
        assert history[0]["success"] is False

        proc0_after = await mem.get_procedure_by_id("proc_0")
        assert proc0_after["metadata"]["execution_count"] == 1
        assert proc0_after["metadata"]["success_rate"] == 0.0
        assert proc0_after["metadata"]["average_duration"] == 12.0
        assert "(Adapted: Rebuild artifact)" in proc0_after["steps"][0]

        # Record a successful execution -> success_rate updates, no adaptation.
        await mem.record_execution("proc_0", success=True, duration=8.0)
        proc0_after2 = await mem.get_procedure_by_id("proc_0")
        assert proc0_after2["metadata"]["execution_count"] == 2
        assert proc0_after2["metadata"]["success_rate"] == 0.5
        assert proc0_after2["metadata"]["average_duration"] == 10.0

        # Regression check: get_procedure_stats/get_procedure_suggestions used
        # to raise (they called super().get_procedure_stats(), which does not
        # exist on BaseMemory). Confirm both now return real aggregated data.
        stats = await mem.get_procedure_stats()
        assert stats["total_procedures"] == 2
        assert stats["chain_stats"]["total_chains"] == 2
        assert stats["execution_stats"]["total_executions"] == 2

        suggestions = await mem.get_procedure_suggestions()
        assert isinstance(suggestions, list)

        await mem.clear()
        assert await mem.get_messages() == []
        assert await mem.get_procedure_by_id("proc_0") is None

    @pytest.mark.asyncio
    async def test_save_load_round_trip(self, tmp_path):
        storage_path = str(tmp_path / "procedural.json")
        mem1 = ProceduralMemory(llm=ProceduralFakeLLM(), storage_path=storage_path)
        await mem1.add_message({"content": "How to deploy the service to production"})
        await mem1.record_execution("proc_0", success=True, duration=5.0)

        mem2 = ProceduralMemory(llm=ProceduralFakeLLM(), storage_path=storage_path)
        await mem2.load()

        messages = await mem2.get_messages()
        assert len(messages) == 1
        # Regression check for the missing-await bug in load()'s embedding recreation.
        assert len(mem2.procedure_embeddings) == 1
        assert all(isinstance(v, list) for v in mem2.procedure_embeddings)
        history = await mem2.get_execution_history("proc_0")
        assert len(history) == 1


# ---------------------------------------------------------------------------
# HybridMemory
# ---------------------------------------------------------------------------


class TestHybridMemory:
    @pytest.mark.asyncio
    async def test_routes_across_two_real_child_memories(self, tmp_path):
        mem = HybridMemory(
            llm=HybridFakeLLM(route_to=["TimeWeightedMemory", "KnowledgeGraphMemory"]),
            storage_path=str(tmp_path),
            memory_types=[TimeWeightedMemory, KnowledgeGraphMemory],
            max_memories=2,
        )

        await mem.add_message({"content": "Remember to water the plants"})

        # Both children actually received the message.
        tw_messages = await mem.memories["TimeWeightedMemory"].get_messages()
        kg_messages = await mem.memories["KnowledgeGraphMemory"].get_messages()
        assert len(tw_messages) == 1
        assert tw_messages[0]["content"] == "Remember to water the plants"
        assert len(kg_messages) == 1
        assert kg_messages[0]["content"] == "Remember to water the plants"

        assert mem.memory_configs["TimeWeightedMemory"]["performance"]["hits"] == 1
        assert mem.memory_configs["KnowledgeGraphMemory"]["performance"]["hits"] == 1

        # Aggregated get_messages across children.
        all_messages = await mem.get_messages()
        assert len(all_messages) == 2

        stats = await mem.get_hybrid_stats()
        assert stats["memory_stats"]["total_memories"] == 2
        assert set(stats["memory_stats"]["memory_types"]) == {
            "TimeWeightedMemory",
            "KnowledgeGraphMemory",
        }
        assert stats["memory_stats"]["total_messages"] == 2
        assert stats["routing_stats"]["total_routes"] == 1

        suggestions = await mem.get_hybrid_suggestions()
        assert isinstance(suggestions, list)

        await mem.clear()
        assert await mem.get_messages() == []
        assert mem.memory_configs == {}

    @pytest.mark.asyncio
    async def test_routes_to_single_memory_and_skips_the_other(self, tmp_path):
        mem = HybridMemory(
            llm=HybridFakeLLM(route_to=["KnowledgeGraphMemory"]),
            storage_path=str(tmp_path),
            memory_types=[TimeWeightedMemory, KnowledgeGraphMemory],
            max_memories=2,
        )

        await mem.add_message({"content": "Only goes to the graph"})

        tw_messages = await mem.memories["TimeWeightedMemory"].get_messages()
        kg_messages = await mem.memories["KnowledgeGraphMemory"].get_messages()
        assert tw_messages == []
        assert len(kg_messages) == 1
        assert mem.memory_configs["TimeWeightedMemory"]["performance"]["hits"] == 0
        assert mem.memory_configs["KnowledgeGraphMemory"]["performance"]["hits"] == 1

    @pytest.mark.asyncio
    async def test_save_load_round_trip_restores_children(self, tmp_path):
        storage_path = str(tmp_path)
        mem1 = HybridMemory(
            llm=HybridFakeLLM(route_to=["TimeWeightedMemory", "KnowledgeGraphMemory"]),
            storage_path=storage_path,
            memory_types=[TimeWeightedMemory, KnowledgeGraphMemory],
            max_memories=2,
        )
        await mem1.add_message({"content": "Persisted message"})

        mem2 = HybridMemory(
            llm=HybridFakeLLM(route_to=["TimeWeightedMemory", "KnowledgeGraphMemory"]),
            storage_path=storage_path,
            memory_types=[TimeWeightedMemory, KnowledgeGraphMemory],
            max_memories=2,
        )
        # Freshly constructed children start out empty.
        assert await mem2.get_messages() == []

        await mem2.load()

        # Regression check: load() must also reload each child memory's own
        # persisted content, not just the hybrid bookkeeping structures.
        restored = await mem2.get_messages()
        assert len(restored) == 2
        assert mem2.routing_history and mem2.routing_history[0]["message"]["content"] == (
            "Persisted message"
        )


# ---------------------------------------------------------------------------
# VectorStoreMemory
# ---------------------------------------------------------------------------


def _small_faiss_config(storage_path=None):
    return VectorStoreConfig(
        connection_params={
            "dimension": 4,
            "max_vectors": 1000,
            "storage_path": storage_path,
            "index_type": "flat",
            "metric": "l2",
        },
        store_type=VectorStoreType.FAISS,
    )


class TestVectorStoreMemory:
    @pytest.mark.asyncio
    async def test_constructs_with_default_config(self):
        # Regression check: VectorStoreMemory() used to crash immediately
        # because the default VectorStoreConfig() call used kwargs the real
        # VectorStoreConfig doesn't accept.
        mem = VectorStoreMemory(llm=VectorFakeLLM())
        assert mem.vector_store is not None
        assert mem.vector_store.config.get("dimension") == 1536

    @pytest.mark.asyncio
    async def test_add_get_search_similarity_ordering(self):
        mem = VectorStoreMemory(llm=VectorFakeLLM(), vector_store_config=_small_faiss_config())

        await mem.add("id_apple", "apple pie recipe")
        await mem.add("id_banana", "banana bread recipe")
        await mem.add("id_car", "car engine repair")

        # search() returns real FAISS nearest-neighbor ordering.
        results = await mem.search("apple-like fruit dessert", k=3)
        assert len(results) == 3
        assert results[0].id == "id_apple"
        assert results[-1].id == "id_car"
        # Nearest result must actually be closer (higher score) than the farthest.
        assert results[0].score > results[-1].score

        # add_message()/get_messages() round trip through the same store.
        await mem.add_message({"role": "user", "content": "banana bread recipe"})
        messages = await mem.get_messages()
        contents = [m["content"] for m in messages]
        assert "banana bread recipe" in contents

    @pytest.mark.asyncio
    async def test_get_by_id_single_entry(self):
        # NOTE: the FAISS backend's search() ignores filter_criteria entirely,
        # so VectorStoreMemory.get(memory_id) cannot reliably filter by ID once
        # more than one vector is stored (known backend limitation, out of
        # scope for memory/vector_store.py). With a single stored vector this
        # still returns the right (only) result.
        mem = VectorStoreMemory(llm=VectorFakeLLM(), vector_store_config=_small_faiss_config())
        await mem.add("id_apple", "apple pie recipe")

        result = await mem.get("id_apple")
        assert result is not None
        assert result.id == "id_apple"

    @pytest.mark.asyncio
    async def test_clear_empties_store(self):
        mem = VectorStoreMemory(llm=VectorFakeLLM(), vector_store_config=_small_faiss_config())
        await mem.add("id_apple", "apple pie recipe")
        assert len(await mem.get_messages()) == 1

        await mem.clear()
        assert await mem.get_messages() == []

    @pytest.mark.asyncio
    async def test_save_load_round_trip(self, tmp_path):
        storage_path = str(tmp_path / "vecstore")
        config = _small_faiss_config(storage_path=storage_path)

        mem1 = VectorStoreMemory(llm=VectorFakeLLM(), vector_store_config=config)
        await mem1.add("id_apple", "apple pie recipe")
        await mem1.add("id_banana", "banana bread recipe")
        await mem1.save()

        mem2 = VectorStoreMemory(llm=VectorFakeLLM(), vector_store_config=config)
        await mem2.load()

        messages = await mem2.get_messages()
        contents = {m["content"] for m in messages}
        assert contents == {"apple pie recipe", "banana bread recipe"}

        # Regression check: VectorStore.load() used to discard the backend
        # returned by the backend's own load() classmethod, leaving the
        # reloaded store empty.
        results = await mem2.search("apple-like fruit dessert", k=1)
        assert len(results) == 1
        assert results[0].id == "id_apple"
