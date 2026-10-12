---
title: "Memory Implementations"
---

This document lists the memory classes exported by `multimind.memory`, what state each one is actually in, and how to use the ones that work. Every class below is importable from `multimind.memory`; the lightweight ones are imported eagerly and the rest resolve lazily on first access.

All memory classes derive from `BaseMemory`, whose contract is five async methods: `add_message(message)`, `get_messages()`, `clear()`, `save()` and `load()`. A class that does not implement all five is abstract and raises `TypeError: Can't instantiate abstract class ...` when constructed. Those classes are listed separately below as scaffolds; their constructor signatures and helper methods exist, but you cannot use them without subclassing.

For the authoritative stable/beta/planned status, see [feature-status.md](/feature-status#memory-systems).

## Extras

| Extra | Needed for |
|-------|------------|
| none (core install) | `BufferMemory`, `BufferWindowMemory`, `SummaryMemory`, `SummaryBufferMemory`, `TokenAwareMemory`, `EntityMemory`, `TimeWeightedMemory`, `HierarchicalMemory`, `EpisodicMemory`, `SemanticMemory`, `ProceduralMemory`, `AssociativeMemory`, `EmotionalMemory`, `DeclarativeMemory`, `SpatialMemory`, `TemporalMemory`, `SensoryMemory`, `VersionedMemory`, `EventSourcedMemory`, `CognitiveScratchpadMemory`, `ActiveLearningMemory`, `ConsensusMemory`, `ReinforcementMemory`, `GenerativeMemory`, `ChatMemory`, `SimpleMemory`, `CombinedMemory`, `ReadOnlyMemory`, `SQLAlchemyMemory` |
| `pip install "multimind-sdk[memory]"` (redis, numpy, networkx) | `VectorStoreMemory`, `KnowledgeGraphMemory`, `HybridMemory`, `ContextualMemory`, `WorkingMemory`, `ForgettingCurveMemory`, `NoveltyMemory`, `DNCMemory`, `ImplicitMemory`, `PlanningMemory`, `RedisMemory`, plus the scaffolds that need numpy |
| `pip install "multimind-sdk[finetune]"` (torch) | `FastWeightMemory`, `AdapterMemory`, `FederatedMemory`, `BayesianMemory`, `AdvancedMemory` |

## Implemented Memory Types

These classes implement the full `BaseMemory` contract and can be instantiated directly. Most of the LLM-backed ones take a `BaseLLM` instance (for example `OpenAIModel`) as their first argument and call it for summarization, routing or extraction.

### Conversation buffers (no LLM required)

- `BufferMemory(memory_key="chat_history", max_tokens=2000, max_messages=None, strategy="sliding", ...)` - stores conversation history with sliding/FIFO/LRU pruning
- `BufferWindowMemory(window_size=10, window_type="count", ...)` - sliding window of recent messages
- `SummaryBufferMemory(max_messages=None, max_tokens=None, summary_interval=10, ...)` - buffer plus periodic summaries
- `TokenAwareMemory` - token-budgeted history
- `TimeWeightedMemory(decay_rate=0.1, ...)` - time-decayed relevance
- `EntityMemory(entity_types=None, relationship_types=None, ...)` - entity and relationship tracking
- `ChatMemory`, `SimpleMemory`, `CombinedMemory`, `ReadOnlyMemory` - small utility memories
- `SQLAlchemyMemory`, `RedisMemory` - persistence-backed buffers (`RedisMemory` needs the `memory` extra)

### LLM-backed memories (`llm` is the first constructor argument)

- `SummaryMemory(llm, ...)` - summarized conversation history
- `VectorStoreMemory(llm, vector_store_config=None, similarity_threshold=0.7, ...)` - embedding-based similarity retrieval over a FAISS store by default
- `KnowledgeGraphMemory(llm, ...)` - graph of entities and relations (needs networkx)
- `HybridMemory(llm, memory_types=None, routing_threshold=0.7, ...)` - routes each message to the best of several child memories
- `HierarchicalMemory(llm, ...)`, `ContextualMemory(llm, ...)`, `EpisodicMemory(llm, ...)`, `SemanticMemory(llm, ...)`, `ProceduralMemory(llm, ...)`, `WorkingMemory(llm, ...)`, `AssociativeMemory(llm, ...)`, `EmotionalMemory(llm, ...)`, `DeclarativeMemory(llm, ...)`, `SpatialMemory(llm, ...)`, `TemporalMemory(llm, ...)`, `SensoryMemory(llm, ...)`, `ForgettingCurveMemory(llm, ...)`, `NoveltyMemory(llm, ...)`, `VersionedMemory(llm, ...)`, `EventSourcedMemory(llm, ...)`, `CognitiveScratchpadMemory(llm, ...)`, `ActiveLearningMemory(llm, ...)`, `DNCMemory(llm, ...)`

### Other implemented memories

- `ImplicitMemory(skill_decay=0.95, max_skills=1000, ...)` - skill tracking with prerequisite graph
- `ConsensusMemory(node_id, nodes=None, ...)` - local in-process replication with majority-vote reads; networked RAFT (elections, RPCs) is not implemented and raises `NotImplementedError`
- `ReinforcementMemory(total_budget=1000, ...)` - tabular Q-learning for keep/evict/compress budgeting
- `PlanningMemory(similarity_threshold=0.8, ...)` - plan-step recording with success-weighted next-step suggestions
- `GenerativeMemory` - regeneration bookkeeping with a caller-supplied generator

## Scaffolds (abstract, cannot be instantiated)

The following classes exist and expose their own methods (for example `add_memory` / `get_memory` on the quantum classes), but they do not implement `add_message`, `get_messages`, `clear`, `save` and/or `load` from `BaseMemory`. Constructing them raises `TypeError`. Treat them as starting points for a subclass, not as usable memories.

| Class | Notes |
|-------|-------|
| `TokenBufferMemory` | token-count buffer; missing `save`/`load` |
| `MetaMemory(base_memory, ...)` | memory about memories |
| `SketchMemory` | count-min sketch / bloom-filter approximate storage |
| `CausalMemory` | causal relationship store |
| `NeuroSymbolicMemory` | neural + symbolic hybrid |
| `AutobiographicalMemory` | self-referential event memory |
| `ProspectiveMemory` | reminders / intentions |
| `ExplicitMemory` | fact store |
| `AdaptiveMemory` | self-tuning context size |
| `HTMMemory(input_size=1024, num_columns=2048, ...)` | hierarchical temporal memory |
| `SpikingMemory` | neuromorphic spiking network with STDP |
| `BayesianMemory` | nonparametric Bayesian clustering (needs torch) |
| `QRAM`, `QAM`, `TopologicalMemory`, `QuantumClassicalHybridMemory` | classical simulations of quantum memory designs, not quantum hardware |
| `FastWeightMemory`, `AdapterMemory`, `FederatedMemory` | torch-based (finetune extra); implement `add_memory`/`get_memory` only |

Nothing in the package implements holographic, DNA or neuromorphic-quantum memory. There are no `ShortTermMemory` or `LongTermMemory` classes; use `BufferMemory` / `BufferWindowMemory` for short-term history and `VectorStoreMemory` or `SQLAlchemyMemory` for durable storage.

## Usage Examples

```python
import asyncio
from multimind import OpenAIModel
from multimind.memory import BufferMemory, VectorStoreMemory, HybridMemory, SummaryMemory

model = OpenAIModel(model_name="gpt-4o-mini")   # reads OPENAI_API_KEY

async def main():
    # Plain conversation buffer (no LLM calls)
    conv_memory = BufferMemory()
    await conv_memory.add_message({"role": "user", "content": "This is an example memory"})
    print(await conv_memory.get_messages())

    # Vector store memory (requires multimind-sdk[memory]; embeds via the model)
    vector_memory = VectorStoreMemory(llm=model)

    # Summary memory (calls the model to summarize older turns)
    summary_memory = SummaryMemory(llm=model)

    # Hybrid memory routes each message to the best child memory (uses the model)
    hybrid_memory = HybridMemory(
        llm=model,
        memory_types=[BufferMemory, VectorStoreMemory],
    )
    await hybrid_memory.add_message({"role": "user", "content": "Remember that I prefer Python."})
    print(await hybrid_memory.get_messages())

asyncio.run(main())
```

Agents use a separate, simpler `multimind.agents.AgentMemory(max_history=100)` for task/response history; see [rag.md](/rag#agent-integration).

## Best Practices

1. **Memory Selection**
   - Start with `BufferMemory`; add `SummaryMemory` or `VectorStoreMemory` only when history outgrows the context window
   - Check the extras table above before importing a memory type
   - Do not rely on the scaffold classes in production code

2. **Memory Configuration**
   - `VectorStoreMemory` defaults to a FAISS backend; the FAISS backend ignores `filter_criteria`, so `get(memory_id)` is not reliable once more than one vector is stored
   - Set `storage_path` when you need `save()`/`load()` persistence

3. **Memory Management**
   - Call `clear()` between unrelated conversations
   - LLM-backed memories make model calls on `add_message`; budget for that cost

## Contributing

To contribute new memory implementations (or to finish one of the scaffolds):

1. Create or edit a file in the `multimind/memory` directory
2. Implement all five `BaseMemory` methods: `add_message`, `get_messages`, `clear`, `save`, `load`
3. Register the class in `multimind/memory/__init__.py` (`_LAZY_ATTRS`, with the extra it needs)
4. Update this document and [feature-status.md](/feature-status)
5. Add tests
6. Submit a pull request
