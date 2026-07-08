"""Governance adapters for LlamaIndex.

* :func:`guard_llm` — PII guard around any llama_index LLM; the result is
  itself an LLM, so it drops into query engines, chat engines, and Settings.
* :class:`MultiMindLlamaIndexHandler` — a llama_index callback handler that
  streams LLM cost/audit events into MultiMind's CostTracker/AuditLog and
  enforces an optional Budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

_INSTALL_HINT = (
    "multimind.integrations.frameworks.llamaindex requires the `llama-index-core` "
    "package. Install with: pip install llama-index-core"
)

try:
    from llama_index.core.base.llms.types import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
    )
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    from llama_index.core.llms import LLM
except ImportError as exc:  # pragma: no cover - exercised via import-safety test
    raise ImportError(_INSTALL_HINT) from exc

from multimind.observability.cost_tracker import _usage_tokens, get_default_tracker

from ._base import GuardState

__all__ = ["GuardedLLM", "MultiMindLlamaIndexHandler", "guard_llm"]


class GuardedLLM(LLM):
    """PII guard around any llama_index LLM; an LLM itself, so it is a drop-in
    replacement anywhere LlamaIndex accepts one.

    Inputs (prompts and string message content) are screened/redacted before
    reaching the wrapped LLM; complete/chat outputs are scanned and redacted;
    streaming variants pass chunks through untouched and scan the accumulated
    text afterwards. Non-string content blocks pass through unscreened.
    """

    _inner: Any = PrivateAttr()
    _state: GuardState = PrivateAttr()

    def __init__(self, inner: Any, state: GuardState, **kwargs: Any):
        super().__init__(**kwargs)
        self._inner = inner
        self._state = state

    @classmethod
    def class_name(cls) -> str:
        return "MultiMindGuardedLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return self._inner.metadata

    # -- sync -----------------------------------------------------------

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        prompt = self._state.screen_input(prompt, "complete")
        response = self._inner.complete(prompt, formatted=formatted, **kwargs)
        return self._screen_completion(response, "complete")

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = self._screen_messages(messages, "chat")
        response = self._inner.chat(messages, **kwargs)
        return self._screen_chat(response, "chat")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        prompt = self._state.screen_input(prompt, "stream_complete")
        inner_gen = self._inner.stream_complete(prompt, formatted=formatted, **kwargs)

        def gen():
            text = ""
            for r in inner_gen:
                text = r.text if isinstance(r.text, str) else text
                yield r
            self._state.scan_output(text, "stream_complete")

        return gen()

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        messages = self._screen_messages(messages, "stream_chat")
        inner_gen = self._inner.stream_chat(messages, **kwargs)

        def gen():
            text = ""
            for r in inner_gen:
                content = r.message.content
                text = content if isinstance(content, str) else text
                yield r
            self._state.scan_output(text, "stream_chat")

        return gen()

    # -- async ----------------------------------------------------------

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        prompt = self._state.screen_input(prompt, "acomplete")
        response = await self._inner.acomplete(prompt, formatted=formatted, **kwargs)
        return self._screen_completion(response, "acomplete")

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages = self._screen_messages(messages, "achat")
        response = await self._inner.achat(messages, **kwargs)
        return self._screen_chat(response, "achat")

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        prompt = self._state.screen_input(prompt, "astream_complete")
        inner_gen = await self._inner.astream_complete(prompt, formatted=formatted, **kwargs)

        async def gen():
            text = ""
            async for r in inner_gen:
                text = r.text if isinstance(r.text, str) else text
                yield r
            self._state.scan_output(text, "astream_complete")

        return gen()

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        messages = self._screen_messages(messages, "astream_chat")
        inner_gen = await self._inner.astream_chat(messages, **kwargs)

        async def gen():
            text = ""
            async for r in inner_gen:
                content = r.message.content
                text = content if isinstance(content, str) else text
                yield r
            self._state.scan_output(text, "astream_chat")

        return gen()

    # -- helpers ---------------------------------------------------------

    def _screen_messages(self, messages: Sequence[ChatMessage], method: str) -> list:
        out = []
        for m in messages:
            if isinstance(m.content, str):
                out.append(
                    ChatMessage(
                        role=m.role,
                        content=self._state.screen_input(m.content, method),
                        additional_kwargs=m.additional_kwargs,
                    )
                )
            else:
                out.append(m)
        return out

    def _screen_completion(self, response: CompletionResponse, method: str) -> CompletionResponse:
        if isinstance(response.text, str):
            response.text = self._state.screen_output(response.text, method)
        return response

    def _screen_chat(self, response: ChatResponse, method: str) -> ChatResponse:
        if isinstance(response.message.content, str):
            response.message = ChatMessage(
                role=response.message.role,
                content=self._state.screen_output(response.message.content, method),
                additional_kwargs=response.message.additional_kwargs,
            )
        return response


def guard_llm(llm: Any, **guard_kwargs: Any) -> GuardedLLM:
    """Wrap a llama_index LLM with MultiMind's PII guard: ``Settings.llm = guard_llm(llm)``."""
    return GuardedLLM(llm, GuardState("llamaindex", **guard_kwargs))


class MultiMindLlamaIndexHandler(BaseCallbackHandler):
    """LlamaIndex callback handler streaming LLM cost/audit events into MultiMind.

    Attach via ``Settings.callback_manager = CallbackManager([handler])``.
    Records every LLM event into a :class:`CostTracker` (provider-reported
    usage from ``response.raw`` when present, chars/4 estimate otherwise),
    enforces an optional :class:`Budget` (raising ``BudgetExceededError``
    before the next LLM call once exceeded), and writes lifecycle events to
    an optional :class:`AuditLog`.

    Observe-only by construction, not by omission:
    ``CallbackManager.on_event_start`` discards each handler's return value,
    so no callback can rewrite the prompt actually sent to the LLM. Since
    redaction is off the table here, ``on_event_start`` instead runs the same
    PII detector used by :func:`guard_llm` over the outgoing prompt and
    records a ``pii_detected`` audit event (types/counts only, never raw
    text) — use :func:`guard_llm` alongside this handler when redaction is
    required.
    """

    def __init__(
        self,
        tracker: Any = None,
        budget: Any = None,
        audit_log: Any = None,
        tag: Optional[str] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._state = GuardState(
            "llamaindex",
            tracker=tracker if tracker is not None else get_default_tracker(),
            budget=budget,
            audit_log=audit_log,
            tag=tag,
            pricing=pricing,
        )
        self._events: Dict[str, Dict[str, Any]] = {}

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type != CBEventType.LLM:
            return event_id
        self._state.check_budget()
        payload = payload or {}
        model = self._model_name(payload)
        input_text = self._input_text(payload)
        self._events[event_id] = {"input": input_text, "model": model}
        self._state.audit({"event": "llm_start", "model": model, "event_id": event_id})
        pii_types = self._state.detect_counts(input_text)
        if pii_types:
            self._state.audit(
                {
                    "event": "pii_detected",
                    "model": model,
                    "event_id": event_id,
                    "pii_types": pii_types,
                    "count": sum(pii_types.values()),
                }
            )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type != CBEventType.LLM:
            return
        run = self._events.pop(event_id, {})
        output_text, usage = self._output(payload or {})
        info = self._state.record_usage(
            run.get("model", "unknown"),
            "llm",
            input_text=run.get("input", ""),
            output_text=output_text,
            usage=usage,
        )
        self._state.audit(
            {
                "event": "llm_end",
                "model": run.get("model", "unknown"),
                "event_id": event_id,
                **(info or {}),
            }
        )

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @staticmethod
    def _input_text(payload: Dict[str, Any]) -> str:
        prompt = payload.get(EventPayload.PROMPT)
        if isinstance(prompt, str):
            return prompt
        messages = payload.get(EventPayload.MESSAGES) or []
        return "\n".join(
            m.content for m in messages if isinstance(getattr(m, "content", None), str)
        )

    @staticmethod
    def _model_name(payload: Dict[str, Any]) -> str:
        serialized = payload.get(EventPayload.SERIALIZED) or {}
        return str(serialized.get("model") or serialized.get("model_name") or "unknown")

    @staticmethod
    def _output(payload: Dict[str, Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
        response = payload.get(EventPayload.COMPLETION) or payload.get(EventPayload.RESPONSE)
        if response is None:
            return "", None
        if isinstance(response, str):
            return response, None
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            message = getattr(response, "message", None)
            content = getattr(message, "content", None)
            text = content if isinstance(content, str) else str(response)
        raw = getattr(response, "raw", None)
        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        return text, _usage_tokens(usage)
