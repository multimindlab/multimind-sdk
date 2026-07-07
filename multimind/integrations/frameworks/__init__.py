"""Framework interoperability adapters: MultiMind governance for other stacks.

Drop-in wrappers that add MultiMind's PII guard, cost/budget tracking, and
audit trail to LangChain, LlamaIndex, CrewAI, AutoGen, Haystack, and raw
OpenAI clients — no migration required. Everything resolves lazily via PEP
562, so importing this package never requires any of those frameworks; a
missing framework raises a friendly ImportError only when its adapter is
actually used (Haystack is the one exception — see ``haystack.py``, which
is duck-typed and never imports ``haystack`` at all).
"""

from __future__ import annotations

import importlib
from typing import Any

# name -> (module path, pip distribution that provides the framework)
_EXPORTS: dict[str, tuple[str, str]] = {
    "GuardedRunnable": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "GuardedTool": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "MultiMindCallbackHandler": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "MultiMindChatModel": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "guard_runnable": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "guard_tool": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "guard_tools": ("multimind.integrations.frameworks.langchain", "langchain-core"),
    "GuardedLLM": ("multimind.integrations.frameworks.llamaindex", "llama-index-core"),
    "MultiMindLlamaIndexHandler": (
        "multimind.integrations.frameworks.llamaindex",
        "llama-index-core",
    ),
    "guard_llm": ("multimind.integrations.frameworks.llamaindex", "llama-index-core"),
    "GuardedCrewLLM": ("multimind.integrations.frameworks.crewai", "crewai"),
    "guard_crew_llm": ("multimind.integrations.frameworks.crewai", "crewai"),
    "guard_openai": ("multimind.integrations.frameworks.openai_client", "openai"),
    "GuardedChatCompletionClient": (
        "multimind.integrations.frameworks.autogen",
        "autogen-core",
    ),
    "guard_autogen_client": ("multimind.integrations.frameworks.autogen", "autogen-core"),
    "GuardedHaystackGenerator": ("multimind.integrations.frameworks.haystack", "haystack-ai"),
    "guard_haystack_generator": ("multimind.integrations.frameworks.haystack", "haystack-ai"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute lookup, cached on the module after first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, dist = _EXPORTS[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if "Install with: pip install" in str(exc):
            raise  # adapter module already gave the specific hint
        raise ImportError(
            f"`{name}` requires the `{dist}` package. Install with: pip install {dist}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
