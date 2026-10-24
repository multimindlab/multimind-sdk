"""Query OpenAI and Claude through the MultiMind SDK model classes.

Demonstrates ``OpenAIModel`` and ``ClaudeModel`` behind the shared ``BaseLLM``
interface: the same ``generate()`` / ``chat()`` calls work for both providers.

Runs offline: with no ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` set it prints
what to configure and exits 0.
"""

import asyncio
import os
from typing import Dict, Tuple, Type

from multimind.core.exceptions import ConfigurationError
from multimind.models import BaseLLM, ClaudeModel, OpenAIModel

# provider -> (model class, default model name, env vars that hold its key)
PROVIDERS: Dict[str, Tuple[Type[BaseLLM], str, Tuple[str, ...]]] = {
    "openai": (OpenAIModel, "gpt-4o-mini", ("OPENAI_API_KEY",)),
    "claude": (ClaudeModel, "claude-sonnet-4-5", ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")),
}


def build_models() -> Dict[str, BaseLLM]:
    """Instantiate every provider whose API key is present in the environment."""
    models: Dict[str, BaseLLM] = {}
    for name, (model_cls, default_model, env_vars) in PROVIDERS.items():
        if not any(os.getenv(var) for var in env_vars):
            continue
        try:
            models[name] = model_cls(model_name=default_model)
        except ConfigurationError as exc:
            print(f"[{name}] skipped: {exc}")
    return models


async def query_all(models: Dict[str, BaseLLM], prompt: str) -> Dict[str, str]:
    """Send the same prompt to each model via the unified BaseLLM interface."""
    results: Dict[str, str] = {}
    for name, model in models.items():
        print(f"\n--- {name} ({model.model_name}) ---")
        try:
            results[name] = await model.generate(prompt, temperature=0.7, max_tokens=120)
            print(results[name])
        except Exception as exc:  # network / auth errors from the provider
            results[name] = f"error: {exc}"
            print(f"error: {exc}")
    return results


async def main() -> int:
    models = build_models()
    if not models:
        print("No API keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to run this example.")
        return 0

    print(f"Configured providers: {', '.join(models)}")
    await query_all(models, "In one sentence, what is the MultiMind SDK?")

    # chat() takes a message list and works identically across providers
    first_name, first_model = next(iter(models.items()))
    print(f"\n--- chat() via {first_name} ---")
    reply = await first_model.chat(
        [
            {"role": "system", "content": "You answer in exactly three words."},
            {"role": "user", "content": "Describe the ocean."},
        ],
        max_tokens=20,
    )
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
