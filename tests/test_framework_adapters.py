"""Tests for framework interoperability adapters (multimind.integrations.frameworks)."""

import importlib
import io
import json
import sys
import types
from types import SimpleNamespace

import pytest

from multimind.compliance.guard import ComplianceViolationError
from multimind.observability.cost_tracker import Budget, BudgetExceededError, CostTracker

lc = pytest.importorskip("langchain_core")
li = pytest.importorskip("llama_index.core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult  # noqa: E402
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata  # noqa: E402
from llama_index.core.callbacks import CallbackManager, CBEventType  # noqa: E402
from llama_index.core.callbacks.schema import EventPayload  # noqa: E402
from llama_index.core.llms import ChatMessage as LIChatMessage  # noqa: E402
from llama_index.core.llms import CustomLLM  # noqa: E402
from pydantic import Field  # noqa: E402

from multimind.integrations.frameworks import (  # noqa: E402
    MultiMindCallbackHandler,
    MultiMindChatModel,
    MultiMindLlamaIndexHandler,
    guard_llm,
    guard_openai,
    guard_runnable,
)

EMAIL_TEXT = "Contact alice@example.com about the invoice."
PII_RESPONSE = "Reply to bob@x.io as soon as possible."
CLEAN_RESPONSE = "All good, nothing to report."


def audit_records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


class EchoChatModel(BaseChatModel):
    """Records every message it sees; replies with a fixed response."""

    response: str = CLEAN_RESPONSE
    usage: bool = False
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "echo"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append([m.content for m in messages])
        message = AIMessage(content=self.response)
        if self.usage:
            message.usage_metadata = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append([m.content for m in messages])
        for part in self.response.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=part + " "))


class FakeMultiMindModel:
    model_name = "mm-fake"

    def __init__(self, response="hello from multimind"):
        self.response = response
        self.seen = []

    async def chat(self, messages, **kwargs):
        self.seen.append(messages)
        return self.response


class TestLangChainGuard:
    def test_input_redacted_before_model(self):
        model = EchoChatModel()
        out = guard_runnable(model).invoke(EMAIL_TEXT)
        assert model.seen[0] == ["Contact [EMAIL] about the invoice."]
        assert out.content == CLEAN_RESPONSE

    def test_output_redacted(self):
        out = guard_runnable(EchoChatModel(response=PII_RESPONSE)).invoke("hi")
        assert "[EMAIL]" in out.content
        assert "bob@x.io" not in out.content

    def test_message_list_input_redacted(self):
        model = EchoChatModel()
        guard_runnable(model).invoke([HumanMessage(content=EMAIL_TEXT)])
        assert model.seen[0] == ["Contact [EMAIL] about the invoice."]

    def test_block_on_raises(self):
        with pytest.raises(ComplianceViolationError):
            guard_runnable(EchoChatModel(), block_on=("email",)).invoke(EMAIL_TEXT)

    async def test_ainvoke_redacts(self):
        model = EchoChatModel(response=PII_RESPONSE)
        out = await guard_runnable(model).ainvoke(EMAIL_TEXT)
        assert model.seen[0] == ["Contact [EMAIL] about the invoice."]
        assert "bob@x.io" not in out.content

    def test_stream_passthrough_with_post_scan(self):
        stream = io.StringIO()
        model = EchoChatModel(response=PII_RESPONSE)
        chunks = list(guard_runnable(model, audit_log=stream).stream("hi"))
        # passthrough: streamed chunks are not redacted
        assert "bob@x.io" in "".join(c.content for c in chunks)
        output_events = [r for r in audit_records(stream) if r["direction"] == "output"]
        assert output_events[-1]["pii_types"] == {"email": 1}

    async def test_astream_post_scan(self):
        stream = io.StringIO()
        guarded = guard_runnable(EchoChatModel(response=PII_RESPONSE), audit_log=stream)
        chunks = [c async for c in guarded.astream("hi")]
        assert len(chunks) > 1
        output_events = [r for r in audit_records(stream) if r["direction"] == "output"]
        assert output_events[-1]["pii_types"] == {"email": 1}


class TestMultiMindChatModel:
    def test_invoke(self):
        fake = FakeMultiMindModel()
        out = MultiMindChatModel(model=fake).invoke("hi there")
        assert out.content == "hello from multimind"
        assert fake.seen[0] == [{"role": "user", "content": "hi there"}]

    async def test_ainvoke(self):
        fake = FakeMultiMindModel()
        out = await MultiMindChatModel(model=fake).ainvoke("hi there")
        assert out.content == "hello from multimind"

    def test_guarded_chain(self):
        fake = FakeMultiMindModel()
        guarded = guard_runnable(MultiMindChatModel(model=fake))
        guarded.invoke(EMAIL_TEXT)
        assert fake.seen[0][0]["content"] == "Contact [EMAIL] about the invoice."


class TestLangChainCallbackHandler:
    def test_records_estimated_cost(self):
        tracker = CostTracker()
        handler = MultiMindCallbackHandler(tracker=tracker, pricing={"": 0.00001})
        EchoChatModel().invoke("hello world", config={"callbacks": [handler]})
        (record,) = tracker.records
        assert record.provider == "langchain"
        assert record.estimated is True
        assert record.input_tokens > 0 and record.output_tokens > 0
        assert record.cost > 0

    def test_uses_reported_usage_metadata(self):
        tracker = CostTracker()
        handler = MultiMindCallbackHandler(tracker=tracker)
        EchoChatModel(usage=True).invoke("hello", config={"callbacks": [handler]})
        (record,) = tracker.records
        assert (record.input_tokens, record.output_tokens) == (11, 7)
        assert record.estimated is False
        assert record.unpriced is True  # no pricing given -> $0, never fabricated

    def test_budget_raises_before_next_call(self):
        budget = Budget(max_cost=0.0000001)
        handler = MultiMindCallbackHandler(
            tracker=CostTracker(), budget=budget, pricing={"": 0.00001}
        )
        model = EchoChatModel()
        model.invoke("first call", config={"callbacks": [handler]})
        with pytest.raises(BudgetExceededError):
            model.invoke("second call", config={"callbacks": [handler]})

    def test_audit_events_written(self):
        stream = io.StringIO()
        handler = MultiMindCallbackHandler(tracker=CostTracker(), audit_log=stream)
        EchoChatModel().invoke("hello", config={"callbacks": [handler]})
        events = [r["event"] for r in audit_records(stream)]
        assert events == ["llm_start", "llm_end"]


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class EchoLLM(CustomLLM):
    response: str = CLEAN_RESPONSE
    seen: list = Field(default_factory=list)

    @property
    def metadata(self):
        return LLMMetadata(model_name="echo-llm")

    def complete(self, prompt, formatted=False, **kwargs):
        self.seen.append(prompt)
        return CompletionResponse(text=self.response)

    def stream_complete(self, prompt, formatted=False, **kwargs):
        self.seen.append(prompt)

        def gen():
            text = ""
            for part in self.response.split(" "):
                text += part + " "
                yield CompletionResponse(text=text, delta=part + " ")

        return gen()


class TestLlamaIndexGuard:
    def test_complete_redacts_input_and_output(self):
        inner = EchoLLM(response=PII_RESPONSE)
        response = guard_llm(inner).complete(EMAIL_TEXT)
        assert inner.seen[0] == "Contact [EMAIL] about the invoice."
        assert "[EMAIL]" in response.text and "bob@x.io" not in response.text

    def test_chat_redacts_input(self):
        inner = EchoLLM()
        guard_llm(inner).chat([LIChatMessage(role="user", content=EMAIL_TEXT)])
        assert "alice@example.com" not in inner.seen[0]
        assert "[EMAIL]" in inner.seen[0]

    def test_block_on_raises(self):
        with pytest.raises(ComplianceViolationError):
            guard_llm(EchoLLM(), block_on=("email",)).complete(EMAIL_TEXT)

    def test_stream_passthrough_with_post_scan(self):
        stream = io.StringIO()
        inner = EchoLLM(response=PII_RESPONSE)
        responses = list(guard_llm(inner, audit_log=stream).stream_complete("hi"))
        assert "bob@x.io" in responses[-1].text  # passthrough, not redacted
        output_events = [r for r in audit_records(stream) if r["direction"] == "output"]
        assert output_events[-1]["pii_types"] == {"email": 1}

    async def test_acomplete_redacts(self):
        inner = EchoLLM(response=PII_RESPONSE)
        response = await guard_llm(inner).acomplete(EMAIL_TEXT)
        assert inner.seen[0] == "Contact [EMAIL] about the invoice."
        assert "bob@x.io" not in response.text

    def test_metadata_delegates(self):
        assert guard_llm(EchoLLM()).metadata.model_name == "echo-llm"


class TestLlamaIndexHandler:
    def test_records_cost_from_raw_usage(self):
        tracker = CostTracker()
        handler = MultiMindLlamaIndexHandler(tracker=tracker, pricing={"echo": 0.00001})
        manager = CallbackManager([handler])
        payload = {EventPayload.PROMPT: "hello", EventPayload.SERIALIZED: {"model": "echo-llm"}}
        with manager.event(CBEventType.LLM, payload=payload) as event:
            event.on_end(
                payload={
                    EventPayload.COMPLETION: CompletionResponse(
                        text="world",
                        raw={"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                    )
                }
            )
        (record,) = tracker.records
        assert record.model == "echo-llm"
        assert (record.input_tokens, record.output_tokens) == (7, 3)
        assert record.estimated is False
        assert record.cost == pytest.approx(10 * 0.00001)

    def test_budget_raises_before_next_llm_event(self):
        budget = Budget(max_cost=0.0000001)
        handler = MultiMindLlamaIndexHandler(
            tracker=CostTracker(), budget=budget, pricing={"": 0.00001}
        )
        manager = CallbackManager([handler])
        with manager.event(CBEventType.LLM, payload={EventPayload.PROMPT: "one"}) as event:
            event.on_end(payload={EventPayload.COMPLETION: CompletionResponse(text="reply")})
        with pytest.raises(BudgetExceededError):
            handler.on_event_start(CBEventType.LLM, payload={EventPayload.PROMPT: "two"})

    def test_ignores_non_llm_events(self):
        tracker = CostTracker()
        handler = MultiMindLlamaIndexHandler(tracker=tracker)
        handler.on_event_start(CBEventType.RETRIEVE, payload={})
        handler.on_event_end(CBEventType.RETRIEVE, payload={})
        assert tracker.records == []


# ---------------------------------------------------------------------------
# CrewAI (fake module: crewai must not be installed in the test env)
# ---------------------------------------------------------------------------


@pytest.fixture
def crew(monkeypatch):
    mod = types.ModuleType("crewai")

    class FakeBaseLLM:
        def __init__(self, model="fake", temperature=None):
            self.model = model
            self.temperature = temperature
            self.stop = []

    class FakeLLM(FakeBaseLLM):
        def __init__(self, response=CLEAN_RESPONSE, **kw):
            super().__init__(**kw)
            self.response = response
            self.seen = []

        def call(self, messages, tools=None, callbacks=None, available_functions=None):
            self.seen.append(messages)
            return self.response

        async def acall(self, messages, **kwargs):
            self.seen.append(messages)
            return self.response

        def supports_function_calling(self):
            return True

        def get_context_window_size(self):
            return 8192

    mod.BaseLLM = FakeBaseLLM
    mod.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "crewai", mod)
    monkeypatch.delitem(sys.modules, "multimind.integrations.frameworks.crewai", raising=False)
    adapter = importlib.import_module("multimind.integrations.frameworks.crewai")
    yield adapter, mod
    sys.modules.pop("multimind.integrations.frameworks.crewai", None)
    fw = sys.modules.get("multimind.integrations.frameworks")
    if fw is not None:
        fw.__dict__.pop("guard_crew_llm", None)
        fw.__dict__.pop("GuardedCrewLLM", None)


class TestCrewAIGuard:
    def test_string_message_redacted(self, crew):
        adapter, mod = crew
        inner = mod.LLM(response=PII_RESPONSE)
        out = adapter.guard_crew_llm(inner).call(EMAIL_TEXT)
        assert inner.seen[0] == "Contact [EMAIL] about the invoice."
        assert "[EMAIL]" in out and "bob@x.io" not in out

    def test_dict_messages_redacted(self, crew):
        adapter, mod = crew
        inner = mod.LLM()
        adapter.guard_crew_llm(inner).call([{"role": "user", "content": EMAIL_TEXT}])
        assert inner.seen[0][0]["content"] == "Contact [EMAIL] about the invoice."

    def test_is_crew_base_llm_and_delegates(self, crew):
        adapter, mod = crew
        guarded = adapter.guard_crew_llm(mod.LLM(model="gpt-fake"))
        assert isinstance(guarded, mod.BaseLLM)
        assert guarded.model == "gpt-fake"
        assert guarded.supports_function_calling() is True
        assert guarded.get_context_window_size() == 8192
        assert guarded.response == CLEAN_RESPONSE  # arbitrary attr proxied

    def test_budget_raises_mid_crew(self, crew):
        adapter, mod = crew
        tracker = CostTracker()
        budget = Budget(max_cost=0.0000001)
        guarded = adapter.guard_crew_llm(
            mod.LLM(model="gpt-fake"), tracker=tracker, budget=budget, pricing={"gpt": 0.00001}
        )
        guarded.call("first task step")
        with pytest.raises(BudgetExceededError):
            guarded.call("second task step")
        (record,) = tracker.records
        assert record.model == "gpt-fake" and record.provider == "crewai"

    async def test_acall_redacted(self, crew):
        adapter, mod = crew
        inner = mod.LLM()
        await adapter.guard_crew_llm(inner).acall([{"role": "user", "content": EMAIL_TEXT}])
        assert inner.seen[0][0]["content"] == "Contact [EMAIL] about the invoice."

    def test_block_on_raises(self, crew):
        adapter, mod = crew
        with pytest.raises(ComplianceViolationError):
            adapter.guard_crew_llm(mod.LLM(), block_on=("email",)).call(EMAIL_TEXT)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


def _chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, content=CLEAN_RESPONSE, chunks=None, usage=True):
        self.content = content
        self.chunks = chunks or []
        self.usage = usage
        self.calls = []

    def _response(self, kwargs):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5) if self.usage else None
        return SimpleNamespace(
            model=kwargs.get("model"),
            usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self._response(kwargs)


class FakeAsyncCompletions(FakeCompletions):
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            items = list(self.chunks)

            class _AIter:
                def __init__(self):
                    self._it = iter(items)

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    try:
                        return next(self._it)
                    except StopIteration:
                        raise StopAsyncIteration

            return _AIter()
        return self._response(kwargs)


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class TestOpenAIGuard:
    def test_request_and_response_redacted(self):
        fake = FakeCompletions(content=PII_RESPONSE)
        client = guard_openai(_client(fake))
        result = client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": EMAIL_TEXT}]
        )
        assert fake.calls[0]["messages"][0]["content"] == "Contact [EMAIL] about the invoice."
        assert "bob@x.io" not in result.choices[0].message.content

    def test_records_exact_usage(self):
        tracker = CostTracker()
        fake = FakeCompletions()
        client = guard_openai(_client(fake), tracker=tracker, pricing={"gpt": 0.00001})
        client.chat.completions.create(
            model="gpt-test", messages=[{"role": "user", "content": "hi"}]
        )
        (record,) = tracker.records
        assert record.provider == "openai" and record.model == "gpt-test"
        assert (record.input_tokens, record.output_tokens) == (10, 5)
        assert record.estimated is False
        assert record.cost == pytest.approx(15 * 0.00001)

    def test_stream_passthrough_and_post_record(self):
        stream = io.StringIO()
        tracker = CostTracker()
        fake = FakeCompletions(chunks=[_chunk("hel"), _chunk("lo bob@x.io")])
        client = guard_openai(_client(fake), tracker=tracker, audit_log=stream)
        chunks = list(
            client.chat.completions.create(
                model="gpt-test", messages=[{"role": "user", "content": "hi"}], stream=True
            )
        )
        assert [c.choices[0].delta.content for c in chunks] == ["hel", "lo bob@x.io"]
        (record,) = tracker.records
        assert record.estimated is True and record.method.endswith(":stream")
        output_events = [r for r in audit_records(stream) if r["direction"] == "output"]
        assert output_events[-1]["pii_types"] == {"email": 1}

    def test_budget_raises(self):
        budget = Budget(max_cost=0.0000001)
        client = guard_openai(
            _client(FakeCompletions()),
            tracker=CostTracker(),
            budget=budget,
            pricing={"": 0.001},
        )
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "one"}])
        with pytest.raises(BudgetExceededError):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "two"}])

    def test_block_on_raises(self):
        client = guard_openai(_client(FakeCompletions()), block_on=("email",))
        with pytest.raises(ComplianceViolationError):
            client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": EMAIL_TEXT}]
            )

    def test_no_double_wrap(self):
        tracker = CostTracker()
        client = _client(FakeCompletions())
        guard_openai(client, tracker=tracker)
        guard_openai(client, tracker=tracker)
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert len(tracker.records) == 1

    async def test_async_client(self):
        tracker = CostTracker()
        fake = FakeAsyncCompletions(content=PII_RESPONSE)
        client = guard_openai(_client(fake), tracker=tracker)
        result = await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": EMAIL_TEXT}]
        )
        assert fake.calls[0]["messages"][0]["content"] == "Contact [EMAIL] about the invoice."
        assert "bob@x.io" not in result.choices[0].message.content
        assert len(tracker.records) == 1

    async def test_async_stream(self):
        tracker = CostTracker()
        fake = FakeAsyncCompletions(chunks=[_chunk("a "), _chunk("b")])
        client = guard_openai(_client(fake), tracker=tracker)
        stream = await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
        chunks = [c async for c in stream]
        assert len(chunks) == 2
        assert len(tracker.records) == 1


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def _hide_modules(monkeypatch, *prefixes):
    for name in list(sys.modules):
        if name.startswith(prefixes):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for prefix in prefixes:
        monkeypatch.setitem(sys.modules, prefix, None)


class TestImportSafety:
    def test_package_imports_without_any_framework(self, monkeypatch):
        _hide_modules(monkeypatch, "langchain_core", "llama_index", "crewai")
        monkeypatch.delitem(
            sys.modules, "multimind.integrations.frameworks.langchain", raising=False
        )
        monkeypatch.delitem(
            sys.modules, "multimind.integrations.frameworks.llamaindex", raising=False
        )
        fw = importlib.import_module("multimind.integrations.frameworks")
        cached = {name: fw.__dict__.pop(name) for name in fw._EXPORTS if name in fw.__dict__}

        for attr, needle in (
            ("guard_runnable", "pip install langchain-core"),
            ("MultiMindChatModel", "pip install langchain-core"),
            ("guard_llm", "pip install llama-index-core"),
            ("MultiMindLlamaIndexHandler", "pip install llama-index-core"),
            ("guard_crew_llm", "pip install crewai"),
        ):
            with pytest.raises(ImportError, match=needle.replace("pip install ", "")):
                getattr(fw, attr)

        # openai adapter is pure duck-typing; loads even with everything hidden
        assert callable(fw.guard_openai)
        fw.__dict__.pop("guard_openai", None)
        fw.__dict__.update(cached)  # restore for later tests

    def test_unknown_attribute_raises_attribute_error(self):
        fw = importlib.import_module("multimind.integrations.frameworks")
        with pytest.raises(AttributeError):
            fw.does_not_exist

    def test_all_exports_resolve_when_frameworks_present(self, crew):
        fw = importlib.import_module("multimind.integrations.frameworks")
        for name in fw.__all__:
            assert getattr(fw, name) is not None
        fw.__dict__.pop("guard_crew_llm", None)
        fw.__dict__.pop("GuardedCrewLLM", None)
