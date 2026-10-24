import pytest

from multimind.memory.buffer import BufferMemory
from multimind.memory.summary_buffer import SummaryBufferMemory


class StubLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, *args, **kwargs):
        self.calls += 1
        return f"summary {self.calls}"


@pytest.mark.asyncio
async def test_buffer_memory_basic():
    mem = BufferMemory(max_messages=2)
    for content in ("a", "b", "c"):
        await mem.add_message({"role": "user", "content": content})
    # Should only keep last 2
    assert [m["content"] for m in await mem.get_messages()] == ["b", "c"]


@pytest.mark.asyncio
async def test_summary_buffer_memory_basic():
    llm = StubLLM()
    mem = SummaryBufferMemory(llm, max_messages=2, summary_interval=1, buffer_strategy="fixed")
    for i in range(3):
        await mem.add_message({"role": "user", "content": f"sentence {i}"})

    # Each message triggered a summary via the stub LLM.
    assert [s["content"] for s in mem.get_summaries()] == ["summary 1", "summary 2", "summary 3"]
    assert [m["content"] for m in await mem.get_messages()] == [
        "summary 1",
        "summary 2",
        "summary 3",
    ]
    # Fixed buffer keeps only the last max_messages raw messages.
    assert [m["content"] for m in await mem.get_buffer_messages()] == ["sentence 1", "sentence 2"]


@pytest.mark.asyncio
async def test_summary_buffer_memory_token_budget_without_tokenizer():
    mem = SummaryBufferMemory(StubLLM(), max_tokens=3)
    await mem.add_message({"role": "user", "content": "one two"})
    await mem.add_message({"role": "user", "content": "three four"})
    stats = await mem.get_buffer_stats()
    assert stats["message_count"] == 1
    assert stats["token_count"] == 2
