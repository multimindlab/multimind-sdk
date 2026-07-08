"""Seamless model switching: a conversation session that survives provider changes.

``ModelSession`` wraps any BaseLLM-compatible model (duck-typed async ``chat``),
maintains conversation history, and can move the conversation to a different
model mid-flight via ``switch()`` without losing knowledge.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..context_transfer.adapters import AdapterFactory
from ..context_transfer.manager import ContextTransferManager

logger = logging.getLogger(__name__)

_STRATEGIES = ("full", "summary", "adapted")

_SUMMARY_PROMPT = (
    "Summarize the following conversation so another AI assistant can continue "
    "it seamlessly. Preserve every fact, name, decision, constraint, and open "
    "question. Be compact but lose nothing important.\n\nConversation:\n{transcript}"
)


def _model_name(model: Any) -> str:
    name = getattr(model, "model_name", None)
    return name if isinstance(name, str) and name else type(model).__name__


def _transcript(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)


def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
    # Rough heuristic: ~4 characters per token.
    return sum(len(m.get("content", "")) for m in messages) // 4


class ModelSession:
    """Stateful chat session over any BaseLLM-compatible model.

    Args:
        model: Model exposing ``async chat(messages, **kwargs) -> str``
            (works with BaseLLM subclasses and wrappers such as ComplianceGuard).
        system_prompt: Optional system message placed at the start of history.
        max_history_tokens: When set, older messages are automatically
            summarized into a rolling summary once the estimated token count
            of the history exceeds this limit.
        keep_recent: Number of most recent messages preserved verbatim when
            the rolling summary compacts history.
    """

    def __init__(
        self,
        model: Any,
        system_prompt: Optional[str] = None,
        max_history_tokens: Optional[int] = None,
        keep_recent: int = 4,
    ):
        self.model = model
        self.max_history_tokens = max_history_tokens
        self.keep_recent = max(1, keep_recent)
        self.transfers: List[Dict[str, Any]] = []
        self._manager = ContextTransferManager()
        self._history: List[Dict[str, str]] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    @property
    def history(self) -> List[Dict[str, str]]:
        """Copy of the current conversation history."""
        return [dict(m) for m in self._history]

    async def chat(self, prompt: str, **kwargs) -> str:
        """Send a user message; history is maintained across calls."""
        self._history.append({"role": "user", "content": prompt})
        await self._compact_if_needed()
        response = await self.model.chat(self.history, **kwargs)
        self._history.append({"role": "assistant", "content": response})
        return response

    def chat_sync(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper around chat()."""
        coro = self.chat(prompt, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        coro.close()
        raise RuntimeError(
            "chat_sync() cannot be called from a running event loop; "
            "await the async chat() method instead."
        )

    async def switch(
        self,
        new_model: Any,
        strategy: str = "full",
        summarizer: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Move the conversation to ``new_model`` without losing knowledge.

        Strategies:
            "full": raw history is replayed to the new model as-is.
            "summary": current model (or ``summarizer``) compacts the history
                into a summary injected as a system message.
            "adapted": routes through ContextTransferManager's provider
                adapters when the target provider is known, else falls
                back to "full".
        """
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r} (use one of {_STRATEGIES})")

        from_name = _model_name(self.model)
        to_name = _model_name(new_model)
        message_count = len(self._history)
        applied = strategy

        if strategy == "summary" and self._history:
            summary = await self._summarize(self._history, summarizer or self.model, **kwargs)
            self._history = self._rebuild_history(
                f"Context transferred from {from_name}. Conversation summary:\n{summary}"
            )
        elif strategy == "adapted" and self._history:
            try:
                adapter = AdapterFactory.get_adapter(to_name)
            except ValueError:
                logger.info("No adapter for %s; falling back to 'full' history transfer", to_name)
                applied = "full"
            else:
                summary = self._manager.summarize_context(self._history)
                context = adapter.format_context(summary, source_model=from_name)
                self._history = self._rebuild_history(context)

        self.transfers.append(
            {
                "from_model": from_name,
                "to_model": to_name,
                "strategy": applied,
                "message_count": message_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.model = new_model
        return new_model

    async def _summarize(self, messages: List[Dict[str, str]], model: Any, **kwargs) -> str:
        prompt = _SUMMARY_PROMPT.format(transcript=_transcript(messages))
        return await model.chat([{"role": "user", "content": prompt}], **kwargs)

    def _rebuild_history(self, context: str) -> List[Dict[str, str]]:
        # Preserve original system messages, then inject transferred context.
        preserved = [m for m in self._history if m.get("role") == "system"]
        return preserved + [{"role": "system", "content": context}]

    async def _compact_if_needed(self) -> None:
        if self.max_history_tokens is None:
            return
        if _estimate_tokens(self._history) <= self.max_history_tokens:
            return
        if len(self._history) <= self.keep_recent:
            return
        recent = self._history[-self.keep_recent :]
        older = self._history[: -self.keep_recent]
        summary = await self._summarize(older, self.model)
        self._history = [
            {"role": "system", "content": f"Summary of earlier conversation:\n{summary}"}
        ] + recent
        logger.info(
            "Compacted %d messages into rolling summary; %d recent kept",
            len(older),
            len(recent),
        )
