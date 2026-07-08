"""Shared test configuration and fixtures.

Markers themselves are registered in ``pyproject.toml`` under
``[tool.pytest.ini_options].markers`` (single source of truth). Do *not*
re-register them here — duplicating them causes warnings and drift.

This file only provides:
  * fixtures that several tests need (fake API keys, etc.)
  * importable skip decorators for optional dependencies
  * a collection hook that auto-skips ``@pytest.mark.requires_api_key`` tests
    when no real API key is in the environment
"""

from __future__ import annotations

import os

import pytest


# --- collection hook ---------------------------------------------------------

# Any of these env vars being set is treated as "we have at least one key
# available", so the test gets a chance to run. Tests that need a specific
# provider should also gate on that provider's individual env var.
_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``requires_api_key`` tests when no API keys are configured."""
    if any(os.getenv(v) for v in _API_KEY_ENV_VARS):
        return  # at least one key set → let tests run and self-gate
    skip = pytest.mark.skip(
        reason="No API keys in environment (set OPENAI_API_KEY etc. to run)"
    )
    for item in items:
        if "requires_api_key" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set a fake OpenAI key for tests that check key presence but don't call the API."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def mock_anthropic_key(monkeypatch):
    """Set a fake Anthropic key for tests that check key presence but don't call the API."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-for-testing")


@pytest.fixture
def mock_all_api_keys(mock_openai_key, mock_anthropic_key, monkeypatch):
    """Convenience: set fakes for every API key the SDK looks at."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-fake-mistral-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key")
    monkeypatch.setenv("COHERE_API_KEY", "test-fake-cohere-key")
    monkeypatch.setenv("HF_TOKEN", "hf_test_fake_token")


# Skip decorators for tests that hit live APIs.
#
# Use these as decorators on individual tests:
#
#     @requires_openai
#     def test_real_chat_completion():
#         ...
#
# For tests that only need a key string present (no real API call), use the
# ``mock_openai_key`` / ``mock_anthropic_key`` fixtures instead.
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


# Skip decorators for tests that need heavy optional dependencies.
#
# Mirror the extras groups in ``pyproject.toml`` so users get a useful hint
# when a test is skipped:
#
#     pip install multimind-sdk[finetune]
def _try_import(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _preload_sqlite_vss_before_faiss() -> None:
    """Avoid a native symbol collision between ``faiss`` and ``sqlite-vss``.

    ``sqlite-vss`` vendors its own private build of libfaiss inside its
    ``vss0``/``vector0`` extensions. If the ``faiss`` PyPI package's libfaiss
    is loaded into the process *first*, sqlite-vss's virtual-table dispatch
    gets corrupted by the collision and every insert fails with a
    nonsensical ``"add_with_ids not implemented for this type of index"``
    error. Loading sqlite-vss's extension pair into a throwaway connection
    here — before ``HAS_FAISS`` below ever imports ``faiss`` — fixes the
    load order for the whole test session regardless of which tests run.
    """
    try:
        import sqlite3

        import sqlite_vss
    except ImportError:
        return
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        vss_path = sqlite_vss.vss_loadable_path()
        vector0_path = os.path.join(os.path.dirname(vss_path), "vector0")
        try:
            conn.load_extension(vector0_path)
        except sqlite3.OperationalError:
            pass  # already registered, missing, or fused into vss0 on this build
        conn.load_extension(vss_path)
        conn.close()
    except Exception:
        pass  # best-effort; sqlite-vss tests self-skip if the ordering wasn't achieved


_preload_sqlite_vss_before_faiss()

HAS_TORCH = _try_import("torch")
HAS_TRANSFORMERS = _try_import("transformers")
HAS_FAISS = _try_import("faiss")
HAS_CHROMADB = _try_import("chromadb")
HAS_PEFT = _try_import("peft")
HAS_PLOTLY = _try_import("plotly")
HAS_CRYPTO_ZKP = _try_import("cryptography.zkp")

requires_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch not installed — install with: pip install multimind-sdk[finetune]",
)

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed — install with: pip install multimind-sdk[finetune]",
)

requires_peft = pytest.mark.skipif(
    not HAS_PEFT,
    reason="peft not installed — install with: pip install multimind-sdk[finetune]",
)

requires_faiss = pytest.mark.skipif(
    not HAS_FAISS,
    reason="faiss-cpu not installed — install with: pip install multimind-sdk[rag]",
)

requires_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB,
    reason="chromadb not installed — install with: pip install multimind-sdk[vector-stores]",
)

requires_plotly = pytest.mark.skipif(
    not HAS_PLOTLY,
    reason="plotly not installed — install with: pip install multimind-sdk[compliance]",
)
