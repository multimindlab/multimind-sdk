"""Route requests across providers with the SDK's ``MultiModelWrapper``.

``MultiModelWrapper`` wraps a ``ModelFactory``: it picks a primary provider,
falls back to the others when a call fails, and tracks per-model performance.

Runs offline: with no provider keys set (and no Ollama server running) it
prints what to configure and exits 0.
"""

import asyncio

from multimind.models import ModelFactory, MultiModelWrapper


def build_wrapper(factory: ModelFactory, primary: str, fallbacks: list) -> MultiModelWrapper:
    return MultiModelWrapper(
        model_factory=factory,
        primary_model=primary,
        fallback_models=fallbacks,
        model_weights={name: 1.0 for name in [primary, *fallbacks]},
        auto_optimize=True,
    )


async def main() -> int:
    factory = ModelFactory()
    available = factory.available_models()
    if not available:
        print(
            "No model providers available. Set OPENAI_API_KEY (or CLAUDE_API_KEY, "
            "GROQ_API_KEY, ...) or start a local Ollama server to run this example."
        )
        return 0

    primary, fallbacks = available[0], available[1:]
    print(f"Primary: {primary}; fallbacks: {fallbacks or 'none'}")
    wrapper = build_wrapper(factory, primary, fallbacks)
    print(f"Initialised models: {list(wrapper.models)}")

    try:
        prompt = "Name one benefit of routing requests across several LLM providers."
        print("\n--- generate() ---")
        print(await wrapper.generate(prompt, temperature=0.5, max_tokens=80))

        print("\n--- chat() ---")
        print(
            await wrapper.chat(
                [{"role": "user", "content": "Reply with the single word: routed"}],
                max_tokens=10,
            )
        )

        print("\n--- performance ---")
        for name, stats in wrapper.get_model_performance().items():
            print(
                f"{name}: score={stats['performance_score']:.3f} "
                f"success_rate={stats['success_rate']:.2f} "
                f"avg_response_time={stats['avg_response_time']:.2f}s"
            )
    finally:
        # OllamaModel keeps an aiohttp session open; release it on exit.
        for model in wrapper.models.values():
            close = getattr(model, "close", None)
            if close is not None:
                await close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
