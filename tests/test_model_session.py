"""Tests for ModelSession (multimind.client.model_session) and the adapter alias fix."""

import pytest

from multimind.client import ModelSession
from multimind.compliance.guard import guard
from multimind.context_transfer import AdapterFactory


class MockModel:
    """Minimal BaseLLM-compatible async model recording every chat call."""

    def __init__(self, name="mock-model", response="ok"):
        self.model_name = name
        self.response = response
        self.chat_calls = []

    async def chat(self, messages, **kwargs):
        self.chat_calls.append([dict(m) for m in messages])
        return self.response


# --- adapter alias regression -------------------------------------------------


def test_hyphenated_aliases_resolve():
    # Regression: "gpt-4"/"claude-3" are registered with hyphens but lookup
    # normalized only the query, so these raised ValueError.
    for alias in ("gpt-4", "gpt-3", "claude-3", "claude-2", "claude-1"):
        adapter = AdapterFactory.get_adapter(alias)
        assert adapter is not None


def test_every_registered_alias_resolves():
    # The CLI --list_models scenario: every supported name must round-trip.
    for name in AdapterFactory.get_supported_models():
        AdapterFactory.get_adapter(name)


def test_alias_separator_variants_resolve():
    assert type(AdapterFactory.get_adapter("gpt_4")) is type(AdapterFactory.get_adapter("gpt-4"))
    assert type(AdapterFactory.get_adapter("GPT 4")) is type(AdapterFactory.get_adapter("gpt-4"))
    assert type(AdapterFactory.get_adapter("anthropic-claude")) is type(
        AdapterFactory.get_adapter("anthropic_claude")
    )


def test_unknown_model_still_raises():
    with pytest.raises(ValueError):
        AdapterFactory.get_adapter("definitely-not-a-model")


def test_list_all_capabilities_covers_all_aliases():
    caps = AdapterFactory.list_all_capabilities()
    assert set(caps) == set(AdapterFactory.get_supported_models())


# --- history accumulation ------------------------------------------------------


async def test_chat_accumulates_history():
    model = MockModel(response="hi there")
    session = ModelSession(model)
    await session.chat("hello")
    await session.chat("how are you?")
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "hi there"},
    ]
    # Second call must have received the full prior history.
    assert model.chat_calls[1][:3] == session.history[:3]


async def test_system_prompt_leads_history():
    session = ModelSession(MockModel(), system_prompt="be terse")
    await session.chat("hi")
    assert session.history[0] == {"role": "system", "content": "be terse"}


def test_history_property_returns_copy():
    session = ModelSession(MockModel())
    session.history.append({"role": "user", "content": "tamper"})
    assert session.history == []


def test_chat_sync():
    model = MockModel(response="sync-reply")
    session = ModelSession(model)
    assert session.chat_sync("hello") == "sync-reply"
    assert len(session.history) == 2


# --- switch strategies ----------------------------------------------------------


async def test_switch_full_replays_raw_history():
    model_a = MockModel(name="model-a", response="a-reply")
    model_b = MockModel(name="model-b", response="b-reply")
    session = ModelSession(model_a)
    await session.chat("remember: the code word is falcon")
    history_before = session.history

    await session.switch(model_b, strategy="full")
    assert session.model is model_b
    assert session.history == history_before

    await session.chat("what is the code word?")
    sent = model_b.chat_calls[0]
    assert any("falcon" in m["content"] for m in sent)


async def test_switch_summary_injects_summary_from_old_model():
    model_a = MockModel(name="model-a", response="SUMMARY: code word falcon")
    model_b = MockModel(name="model-b", response="b-reply")
    session = ModelSession(model_a)
    await session.chat("the code word is falcon")

    await session.switch(model_b, strategy="summary")
    # Old model produced the summary.
    summary_request = model_a.chat_calls[-1]
    assert summary_request[0]["role"] == "user"
    assert "Summarize" in summary_request[0]["content"]
    assert "falcon" in summary_request[0]["content"]
    # History replaced with a system message carrying the summary.
    assert len(session.history) == 1
    assert session.history[0]["role"] == "system"
    assert "SUMMARY: code word falcon" in session.history[0]["content"]

    await session.chat("what is the code word?")
    sent = model_b.chat_calls[0]
    assert sent[0]["role"] == "system"
    assert "falcon" in sent[0]["content"]


async def test_switch_summary_with_explicit_summarizer():
    summarizer = MockModel(name="summarizer", response="compact summary")
    model_a = MockModel(name="model-a")
    model_b = MockModel(name="model-b")
    session = ModelSession(model_a)
    await session.chat("hello")

    await session.switch(model_b, strategy="summary", summarizer=summarizer)
    assert len(summarizer.chat_calls) == 1
    assert "compact summary" in session.history[0]["content"]


async def test_switch_summary_preserves_original_system_prompt():
    model_a = MockModel(name="model-a", response="the summary")
    session = ModelSession(model_a, system_prompt="be terse")
    await session.chat("hi")
    await session.switch(MockModel(name="model-b"), strategy="summary")
    assert session.history[0] == {"role": "system", "content": "be terse"}
    assert "the summary" in session.history[1]["content"]


async def test_switch_adapted_uses_provider_adapter():
    model_a = MockModel(name="chatgpt", response="a-reply")
    model_b = MockModel(name="claude", response="b-reply")
    session = ModelSession(model_a)
    await session.chat("the code word is falcon")

    await session.switch(model_b, strategy="adapted")
    assert len(session.history) == 1
    context = session.history[0]["content"]
    # ClaudeAdapter framing plus the summarized conversation content.
    assert "You are Claude" in context
    assert "chatgpt" in context
    assert "falcon" in context
    assert session.transfers[-1]["strategy"] == "adapted"


async def test_switch_adapted_falls_back_to_full_for_unknown_target():
    model_a = MockModel(name="model-a")
    model_b = MockModel(name="totally-unknown-model")
    session = ModelSession(model_a)
    await session.chat("hello")
    history_before = session.history

    await session.switch(model_b, strategy="adapted")
    assert session.history == history_before
    assert session.transfers[-1]["strategy"] == "full"


async def test_switch_rejects_unknown_strategy():
    session = ModelSession(MockModel())
    with pytest.raises(ValueError):
        await session.switch(MockModel(), strategy="teleport")


# --- transfers log ---------------------------------------------------------------


async def test_transfers_log_records_each_switch():
    model_a = MockModel(name="model-a", response="summary text")
    model_b = MockModel(name="model-b")
    model_c = MockModel(name="model-c")
    session = ModelSession(model_a)
    await session.chat("hello")

    await session.switch(model_b, strategy="full")
    await session.switch(model_c, strategy="summary")

    assert len(session.transfers) == 2
    first, second = session.transfers
    assert first["from_model"] == "model-a"
    assert first["to_model"] == "model-b"
    assert first["strategy"] == "full"
    assert first["message_count"] == 2
    assert first["timestamp"]
    assert second["from_model"] == "model-b"
    assert second["to_model"] == "model-c"
    assert second["strategy"] == "summary"


# --- composability with wrappers --------------------------------------------------


async def test_session_works_with_compliance_guard_wrapped_model():
    inner = MockModel(name="guarded-model", response="clean reply")
    guarded = guard(inner)
    session = ModelSession(guarded)

    reply = await session.chat("email me at john@example.com")
    assert reply == "clean reply"
    # Guard redacted the PII before the inner model saw it.
    assert "[EMAIL]" in inner.chat_calls[0][-1]["content"]
    # Session history keeps what the user actually said.
    assert session.history[0]["content"] == "email me at john@example.com"

    # Switch away from a guarded model works too (guard delegates model_name).
    model_b = MockModel(name="model-b")
    await session.switch(model_b, strategy="full")
    assert session.transfers[-1]["from_model"] == "guarded-model"
    assert session.model is model_b


# --- rolling summary (token awareness) ---------------------------------------------


async def test_rolling_summary_triggers_when_over_budget():
    model = MockModel(name="model-a", response="short")
    session = ModelSession(model, max_history_tokens=50, keep_recent=2)

    long_text = "x" * 400  # ~100 estimated tokens, over the 50 budget
    await session.chat(long_text)
    await session.chat("second message")

    # Compaction ran before the second chat call: one summary system message
    # plus the kept recent tail.
    assert session.history[0]["role"] == "system"
    assert session.history[0]["content"].startswith("Summary of earlier conversation:")
    # A summarization request was made to the model itself.
    assert any("Summarize" in call[0]["content"] for call in model.chat_calls)
    # The long message no longer appears verbatim.
    assert all(long_text != m["content"] for m in session.history)


async def test_no_compaction_under_budget():
    model = MockModel()
    session = ModelSession(model, max_history_tokens=10_000)
    await session.chat("hello")
    await session.chat("again")
    assert all(m["role"] != "system" for m in session.history)
    assert len(session.history) == 4
