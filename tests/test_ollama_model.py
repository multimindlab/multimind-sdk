"""Unit tests for OllamaModel / MistralModel with the aiohttp layer mocked.

No real Ollama server is contacted: a fake ``aiohttp.ClientSession`` is
injected via ``model._session`` so ``_get_session`` returns it directly.
"""

import json

import aiohttp
import pytest
from tenacity import wait_none

from multimind.models.ollama import MistralModel, OllamaModel


class FakeResponse:
    """Minimal stand-in for an aiohttp response used as an async context manager."""

    def __init__(self, payload=None, status=200, lines=None):
        self.payload = payload or {}
        self.status = status
        self._lines = lines or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None, history=(), status=self.status, message="error"
            )

    async def json(self):
        return self.payload

    @property
    def content(self):
        async def _iter():
            for line in self._lines:
                yield line

        return _iter()


class FakeSession:
    """Records every post() call and replays queued responses."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []
        self.closed = False

    def post(self, url, json=None):
        self.calls.append({"url": url, "json": json})
        if self.error is not None:
            raise self.error
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    async def close(self):
        self.closed = True


def make_model(responses=None, error=None, cls=OllamaModel, **kwargs):
    model = cls(**kwargs) if kwargs else cls(model_name="llama3")
    session = FakeSession(responses=responses, error=error)
    model._session = session
    return model, session


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Disable tenacity backoff so the connection-error path runs instantly."""
    monkeypatch.setattr(OllamaModel._make_request_with_retry.retry, "wait", wait_none())
    monkeypatch.setattr(OllamaModel._make_request_stream_raw.retry, "wait", wait_none())


def test_init_defaults_and_base_url_strip():
    model = OllamaModel(model_name="llama3", base_url="http://ollama.local:11434/")
    assert model.base_url == "http://ollama.local:11434"
    assert model.cost_per_token == 0.0
    assert model.avg_latency == 0.1
    assert model._session is None


def test_mistral_model_defaults_and_overrides():
    assert MistralModel().model_name == "mistral"
    assert MistralModel(model="mistral:7b").model_name == "mistral:7b"
    # model_name wins over model when both are given
    assert MistralModel(model="x", model_name="mistral-nemo").model_name == "mistral-nemo"
    assert isinstance(MistralModel(), OllamaModel)


async def test_generate_builds_payload_and_parses_response():
    model, session = make_model([FakeResponse({"response": "hello world", "done": True})])
    out = await model.generate("Say hi", temperature=0.2, max_tokens=50, top_p=0.9)

    assert out == "hello world"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "http://localhost:11434/api/generate"
    # Ollama reads sampling settings from `options` (num_predict, not max_tokens)
    # and streams NDJSON unless stream=false is sent explicitly.
    assert call["json"] == {
        "model": "llama3",
        "prompt": "Say hi",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 50},
        "top_p": 0.9,
    }


async def test_generate_omits_max_tokens_when_none_and_handles_missing_response():
    model, session = make_model([FakeResponse({})])
    assert await model.generate("x") == ""
    assert session.calls[0]["json"]["options"] == {"temperature": 0.7}


async def test_caller_options_are_merged_not_overwritten():
    model, session = make_model([FakeResponse({"response": "ok"})])
    await model.generate("x", temperature=0.1, options={"temperature": 0.9, "seed": 1})
    assert session.calls[0]["json"]["options"] == {"temperature": 0.9, "seed": 1}


async def test_chat_builds_payload_and_parses_message():
    model, session = make_model(
        [FakeResponse({"message": {"role": "assistant", "content": "pong"}})]
    )
    messages = [{"role": "user", "content": "ping"}]
    out = await model.chat(messages, temperature=0.5, max_tokens=10)

    assert out == "pong"
    call = session.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"] == {
        "model": "llama3",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 10},
    }


async def test_chat_returns_empty_string_when_message_missing():
    model, _ = make_model([FakeResponse({"done": True})])
    assert await model.chat([{"role": "user", "content": "x"}]) == ""


async def test_embeddings_single_string_returns_flat_vector():
    model, session = make_model([FakeResponse({"embedding": [0.1, 0.2, 0.3]})])
    out = await model.embeddings("hello")

    assert out == [0.1, 0.2, 0.3]
    call = session.calls[0]
    assert call["url"] == "http://localhost:11434/api/embeddings"
    assert call["json"] == {"model": "llama3", "prompt": "hello"}


async def test_embeddings_list_makes_one_request_per_text():
    model, session = make_model(
        [FakeResponse({"embedding": [1.0, 0.0]}), FakeResponse({"embedding": [0.0, 1.0]})]
    )
    out = await model.embeddings(["a", "b"])

    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert [c["json"]["prompt"] for c in session.calls] == ["a", "b"]


async def test_generate_stream_yields_response_chunks():
    lines = [
        json.dumps({"response": "Hel"}).encode() + b"\n",
        b"",  # blank keep-alive line is skipped
        json.dumps({"response": "lo"}).encode() + b"\n",
        json.dumps({"done": True}).encode() + b"\n",
    ]
    model, session = make_model([FakeResponse(lines=lines)])
    chunks = [c async for c in model.generate_stream("hi")]

    assert chunks == ["Hel", "lo"]
    assert session.calls[0]["json"]["stream"] is True
    assert session.calls[0]["url"].endswith("/api/generate")


async def test_chat_stream_yields_message_content():
    lines = [
        json.dumps({"message": {"content": "a"}}).encode() + b"\n",
        json.dumps({"message": {"content": "b"}}).encode() + b"\n",
        json.dumps({"done": True}).encode() + b"\n",
    ]
    model, session = make_model([FakeResponse(lines=lines)])
    chunks = [c async for c in model.chat_stream([{"role": "user", "content": "x"}])]

    assert chunks == ["a", "b"]
    assert session.calls[0]["json"]["stream"] is True
    assert session.calls[0]["url"].endswith("/api/chat")


async def test_connection_error_is_retried_then_raised(no_retry_wait):
    model, session = make_model(error=aiohttp.ClientConnectionError("refused"))
    with pytest.raises(aiohttp.ClientConnectionError):
        await model.generate("hi")
    # stop_after_attempt(3) -> three attempts before giving up
    assert len(session.calls) == 3


async def test_http_error_status_raises(no_retry_wait):
    model, _ = make_model([FakeResponse({"error": "model not found"}, status=404)])
    with pytest.raises(aiohttp.ClientResponseError):
        await model.chat([{"role": "user", "content": "x"}])


async def test_close_closes_session_and_resets():
    model, session = make_model([FakeResponse({})])
    await model.close()
    assert session.closed is True
    assert model._session is None
    # Closing again with no session is a no-op
    await model.close()
