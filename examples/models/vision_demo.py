"""Offline vision/multimodal demo.

Shows the exact request payloads MultiMind builds when you pass ``images=``
to ``generate()``. Uses fake in-memory clients — no API keys, no network.
With a real key, drop the fake client lines and the same calls hit the API.
"""

import asyncio
import json
from types import SimpleNamespace

from multimind.models.claude import ClaudeModel
from multimind.models.openai import OpenAIModel

# A tiny valid-magic-number PNG payload (enough for media-type sniffing).
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class _Recorder:
    """Captures create() kwargs and returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.last_kwargs = None

    async def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


def fake_openai_client(recorder):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))


def fake_claude_client(recorder):
    return SimpleNamespace(messages=SimpleNamespace(create=recorder))


async def main():
    images = [
        {"bytes": PNG_BYTES},  # raw bytes, media type sniffed
        {"url": "https://example.com/photo.jpg"},  # remote URL, passed through
    ]

    # --- OpenAI-compatible payload (content-parts with data URLs) ---
    openai_recorder = _Recorder(
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="a demo"))])
    )
    openai_model = OpenAIModel(model_name="gpt-4o", api_key="fake-key")
    openai_model.client = fake_openai_client(openai_recorder)

    await openai_model.generate("What is in this image?", images=images)
    print("OpenAI request messages:")
    print(json.dumps(openai_recorder.last_kwargs["messages"], indent=2)[:1200])

    # --- Anthropic payload (image content blocks) ---
    claude_recorder = _Recorder(
        SimpleNamespace(content=[SimpleNamespace(type="text", text="a demo")])
    )
    claude_model = ClaudeModel(model_name="claude-sonnet-4-5", api_key="fake-key")
    claude_model.client = fake_claude_client(claude_recorder)

    await claude_model.generate("What is in this image?", images=images)
    print("\nClaude request messages:")
    print(json.dumps(claude_recorder.last_kwargs["messages"], indent=2)[:1200])


if __name__ == "__main__":
    asyncio.run(main())
