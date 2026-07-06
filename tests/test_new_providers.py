"""Tests for the OpenAI-compatible providers, sync facade, and structured output.

All tests use mocked clients — no live API calls.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from multimind.core.exceptions import ConfigurationError
from multimind.models.base import BaseLLM
from multimind.models.claude import ClaudeModel
from multimind.models.deepseek import DeepSeekModel
from multimind.models.factory import ModelFactory
from multimind.models.gemini import GeminiModel
from multimind.models.groq import GroqModel
from multimind.models.mistral import MistralAIModel
from multimind.models.openai import OpenAIModel

PROVIDERS = [
    (GroqModel, "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    (MistralAIModel, "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    (GeminiModel, "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    (DeepSeekModel, "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
]

ALL_KEY_VARS = [
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
]


@pytest.fixture
def no_api_keys(monkeypatch):
    for var in ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _mock_client(content: str = "mocked") -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_chat_response(content)))
        ),
        embeddings=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
            )
        ),
    )


class Answer(BaseModel):
    value: str
    score: int


class TestProviderConfig:
    @pytest.mark.parametrize("model_cls,base_url,env_var", PROVIDERS)
    def test_base_url(self, model_cls, base_url, env_var, no_api_keys):
        model = model_cls(model_name="some-model", api_key="test-key")
        assert str(model.client.base_url).rstrip("/") == base_url.rstrip("/")

    @pytest.mark.parametrize("model_cls,base_url,env_var", PROVIDERS)
    def test_api_key_from_env(self, model_cls, base_url, env_var, no_api_keys, monkeypatch):
        monkeypatch.setenv(env_var, "env-test-key")
        model = model_cls(model_name="some-model")
        assert model.client.api_key == "env-test-key"

    @pytest.mark.parametrize("model_cls,base_url,env_var", PROVIDERS)
    def test_missing_key_raises(self, model_cls, base_url, env_var, no_api_keys):
        with pytest.raises(ConfigurationError, match=env_var):
            model_cls(model_name="some-model")

    def test_gemini_google_api_key_fallback(self, no_api_keys, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
        model = GeminiModel(model_name="gemini-2.0-flash")
        assert model.client.api_key == "google-test-key"

    async def test_generate_and_chat_inherited(self, no_api_keys):
        model = GroqModel(model_name="llama-3.3-70b-versatile", api_key="test-key")
        model.client = _mock_client("hello")
        assert await model.generate("hi") == "hello"
        assert await model.chat([{"role": "user", "content": "hi"}]) == "hello"

    async def test_embeddings_not_implemented(self, no_api_keys):
        for model_cls in (GroqModel, DeepSeekModel):
            model = model_cls(model_name="some-model", api_key="test-key")
            with pytest.raises(NotImplementedError):
                await model.embeddings("hello")

    async def test_embeddings_supported(self, no_api_keys):
        for model_cls, expected_model in (
            (MistralAIModel, "mistral-embed"),
            (GeminiModel, "gemini-embedding-001"),
        ):
            model = model_cls(model_name="some-model", api_key="test-key")
            model.client = _mock_client()
            result = await model.embeddings("hello")
            assert result == [0.1, 0.2]
            call_kwargs = model.client.embeddings.create.call_args.kwargs
            assert call_kwargs["model"] == expected_model


class TestFactory:
    @pytest.mark.parametrize(
        "provider,model_cls",
        [
            ("groq", GroqModel),
            ("mistral", MistralAIModel),
            ("gemini", GeminiModel),
            ("deepseek", DeepSeekModel),
        ],
    )
    def test_factory_creates_provider(self, provider, model_cls, no_api_keys):
        factory = ModelFactory()
        model = factory.get_model(provider, api_key="test-key")
        assert isinstance(model, model_cls)


class TestSyncFacade:
    class DummyModel(BaseLLM):
        async def generate(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
            return f"gen:{prompt}"

        async def generate_stream(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
            yield "chunk"

        async def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
            return f"chat:{messages[0]['content']}"

        async def chat_stream(self, messages, temperature=0.7, max_tokens=None, **kwargs):
            yield "chunk"

        async def embeddings(self, text, **kwargs):
            return [1.0, 2.0]

    def test_sync_methods_return_async_result(self):
        model = self.DummyModel("dummy")
        assert model.generate_sync("hi") == "gen:hi"
        assert model.chat_sync([{"role": "user", "content": "hi"}]) == "chat:hi"
        assert model.embeddings_sync("hi") == [1.0, 2.0]

    def test_sync_raises_inside_running_loop(self):
        model = self.DummyModel("dummy")

        async def call_sync():
            model.generate_sync("hi")

        with pytest.raises(RuntimeError, match="await the async"):
            asyncio.run(call_sync())


class TestStructuredOutput:
    async def test_openai_parses_pydantic(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")
        model.client = _mock_client('{"value": "ok", "score": 5}')
        result = await model.generate("hi", response_format=Answer)
        assert isinstance(result, Answer)
        assert result.value == "ok"
        assert result.score == 5
        call_kwargs = model.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()

    async def test_openai_malformed_json_raises(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")
        model.client = _mock_client("not json at all")
        with pytest.raises(ValueError, match="not json at all"):
            await model.generate("hi", response_format=Answer)

    async def test_openai_dict_passthrough(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")
        model.client = _mock_client('{"a": 1}')
        result = await model.chat(
            [{"role": "user", "content": "hi"}], response_format={"type": "json_object"}
        )
        assert result == '{"a": 1}'
        call_kwargs = model.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_openai_default_behavior_unchanged(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key")
        model.client = _mock_client("plain text")
        assert await model.generate("hi") == "plain text"
        call_kwargs = model.client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    async def test_claude_parses_pydantic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        model = ClaudeModel()
        tool_block = SimpleNamespace(type="tool_use", input={"value": "ok", "score": 7})
        model.client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(content=[tool_block]))
            )
        )
        result = await model.generate("hi", response_format=Answer)
        assert isinstance(result, Answer)
        assert result.score == 7
        call_kwargs = model.client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "structured_output"}
        assert call_kwargs["tools"][0]["input_schema"] == Answer.model_json_schema()

    async def test_claude_invalid_tool_input_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        model = ClaudeModel()
        tool_block = SimpleNamespace(type="tool_use", input={"wrong": "shape"})
        model.client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(content=[tool_block]))
            )
        )
        with pytest.raises(ValueError, match="Failed to parse"):
            await model.generate("hi", response_format=Answer)


class TestPricing:
    def test_known_model_pricing(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o-mini", api_key="test-key")
        assert model.cost_per_token == OpenAIModel.MODEL_PRICING["gpt-4o-mini"]

    def test_unknown_model_uses_default(self, no_api_keys):
        model = OpenAIModel(model_name="totally-new-model", api_key="test-key")
        assert model.cost_per_token == OpenAIModel.DEFAULT_COST_PER_TOKEN

    def test_constructor_override(self, no_api_keys):
        model = OpenAIModel(model_name="gpt-4o", api_key="test-key", cost_per_token=0.5)
        assert model.cost_per_token == 0.5
