"""Tests for the optional images parameter on generate/chat.

All tests use mocked clients — no live API calls.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from multimind.models.base import resolve_image, resolve_images
from multimind.models.claude import ClaudeModel
from multimind.models.gemini import GeminiModel
from multimind.models.openai import OpenAIModel

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 8
JPEG_B64 = base64.b64encode(JPEG_BYTES).decode("ascii")


def _openai_mock_client(content: str = "mocked") -> SimpleNamespace:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )


def _claude_mock_client(content: str = "mocked") -> SimpleNamespace:
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=content)])
    return SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=response)))


class TestResolveImage:
    def test_bytes_png_sniffed(self):
        resolved = resolve_image({"bytes": PNG_BYTES})
        assert resolved == {"kind": "base64", "media_type": "image/png", "data": PNG_B64}

    def test_bytes_jpeg_sniffed(self):
        resolved = resolve_image({"bytes": JPEG_BYTES})
        assert resolved == {"kind": "base64", "media_type": "image/jpeg", "data": JPEG_B64}

    def test_path(self, tmp_path):
        path = tmp_path / "img.png"
        path.write_bytes(PNG_BYTES)
        resolved = resolve_image({"path": str(path)})
        assert resolved == {"kind": "base64", "media_type": "image/png", "data": PNG_B64}

    def test_url_passthrough(self):
        resolved = resolve_image({"url": "https://example.com/cat.png"})
        assert resolved == {"kind": "url", "url": "https://example.com/cat.png"}

    def test_explicit_media_type_wins(self):
        resolved = resolve_image({"bytes": b"rawdata", "media_type": "image/webp"})
        assert resolved["media_type"] == "image/webp"

    def test_unknown_media_type_rejected(self):
        with pytest.raises(ValueError, match="media type"):
            resolve_image({"bytes": b"not an image"})

    def test_requires_exactly_one_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            resolve_image({"path": "a.png", "url": "https://x"})
        with pytest.raises(ValueError, match="exactly one"):
            resolve_image({"media_type": "image/png"})

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="dict"):
            resolve_image("cat.png")

    def test_resolve_images_maps(self):
        resolved = resolve_images([{"bytes": PNG_BYTES}, {"url": "https://x/y.jpg"}])
        assert [r["kind"] for r in resolved] == ["base64", "url"]


class TestOpenAIVision:
    @pytest.mark.asyncio
    async def test_generate_with_images_payload(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client("a cat")

        result = await model.generate(
            "What is in this image?",
            images=[{"bytes": PNG_BYTES}, {"url": "https://example.com/dog.jpg"}],
        )

        assert result == "a cat"
        call = model.client.chat.completions.create.call_args
        assert call.kwargs["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                    },
                    {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}},
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_without_images_unchanged(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client()

        await model.generate("hello")

        call = model.client.chat.completions.create.call_args
        assert call.kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert "images" not in call.kwargs

    @pytest.mark.asyncio
    async def test_chat_attaches_to_last_user_message(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client()

        messages = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "describe"},
        ]
        await model.chat(messages, images=[{"bytes": PNG_BYTES}])

        sent = model.client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[1] == {"role": "user", "content": "first"}
        assert sent[3] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
            ],
        }
        # Caller's message list is not mutated
        assert messages[3] == {"role": "user", "content": "describe"}

    @pytest.mark.asyncio
    async def test_chat_without_images_unchanged(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client()

        await model.chat([{"role": "user", "content": "hi"}])

        sent = model.client.chat.completions.create.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_chat_with_images_requires_user_message(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client()

        with pytest.raises(ValueError, match="user message"):
            await model.chat([{"role": "system", "content": "x"}], images=[{"bytes": PNG_BYTES}])

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        class NoVisionModel(OpenAIModel):
            PROVIDER_NAME = "NoVision"
            SUPPORTS_VISION = False

        model = NoVisionModel(model_name="text-only", api_key="test")
        model.client = _openai_mock_client()

        with pytest.raises(NotImplementedError, match="NoVision"):
            await model.generate("hi", images=[{"bytes": PNG_BYTES}])
        with pytest.raises(NotImplementedError, match="NoVision"):
            await model.chat([{"role": "user", "content": "hi"}], images=[{"bytes": PNG_BYTES}])
        model.client.chat.completions.create.assert_not_called()

    def test_gemini_inherits_vision_support(self):
        assert GeminiModel.SUPPORTS_VISION is True

    def test_sync_facade_forwards_images(self):
        model = OpenAIModel(model_name="gpt-4o", api_key="test")
        model.client = _openai_mock_client("sync ok")

        result = model.generate_sync("look", images=[{"bytes": PNG_BYTES}])

        assert result == "sync ok"
        sent = model.client.chat.completions.create.call_args.kwargs["messages"]
        assert sent[0]["content"][1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"


class TestClaudeVision:
    @pytest.mark.asyncio
    async def test_generate_with_images_payload(self):
        model = ClaudeModel(model_name="claude-3-opus-20240229", api_key="test")
        model.client = _claude_mock_client("a cat")

        result = await model.generate(
            "What is in this image?",
            images=[{"bytes": PNG_BYTES}, {"url": "https://example.com/dog.jpg"}],
        )

        assert result == "a cat"
        call = model.client.messages.create.call_args
        assert call.kwargs["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": PNG_B64,
                        },
                    },
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/dog.jpg"},
                    },
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]

    @pytest.mark.asyncio
    async def test_generate_without_images_unchanged(self):
        model = ClaudeModel(model_name="claude-3-opus-20240229", api_key="test")
        model.client = _claude_mock_client()

        await model.generate("hello")

        call = model.client.messages.create.call_args
        assert call.kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert "images" not in call.kwargs

    @pytest.mark.asyncio
    async def test_chat_attaches_to_last_user_message(self):
        model = ClaudeModel(model_name="claude-3-opus-20240229", api_key="test")
        model.client = _claude_mock_client()

        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "describe"},
        ]
        await model.chat(messages, images=[{"bytes": PNG_BYTES}])

        sent = model.client.messages.create.call_args.kwargs["messages"]
        assert sent[0] == {"role": "user", "content": "first"}
        assert sent[2]["content"] == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
            },
            {"type": "text", "text": "describe"},
        ]
        assert messages[2] == {"role": "user", "content": "describe"}

    @pytest.mark.asyncio
    async def test_chat_without_images_unchanged(self):
        model = ClaudeModel(model_name="claude-3-opus-20240229", api_key="test")
        model.client = _claude_mock_client()

        await model.chat([{"role": "user", "content": "hi"}])

        sent = model.client.messages.create.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": "hi"}]
