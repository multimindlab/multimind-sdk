"""Governance adapter for raw OpenAI clients.

:func:`guard_openai` patches ``client.chat.completions.create`` on an
``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance in place, so existing
call sites get PII redaction, budget enforcement, cost tracking, and audit
logging with no other change. Return types are preserved: the completion
object is returned as-is (with redacted message content), and streaming
returns a thin pass-through wrapper that scans/records after the last chunk.

Duck-typed on purpose — works against any client object whose
``chat.completions.create`` follows the OpenAI signature (Azure OpenAI,
compatible proxies, test fakes).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Dict

from multimind.observability.cost_tracker import _usage_tokens

from ._base import GuardState

__all__ = ["guard_openai"]

_METHOD = "chat.completions.create"


def guard_openai(client: Any, **guard_kwargs: Any) -> Any:
    """Guard an OpenAI client in place: ``client = guard_openai(OpenAI(), budget=Budget(5.0))``.

    Accepts the shared governance kwargs (``redact_input``, ``redact_output``,
    ``strategy``, ``block_on``, ``detector``, ``audit_log``, ``tracker``,
    ``budget``, ``tag``, ``pricing``). Returns the same client instance.
    """
    state = GuardState("openai", **guard_kwargs)
    completions = client.chat.completions
    if getattr(completions, "_multimind_guarded", False):
        return client
    original = completions.create
    if inspect.iscoroutinefunction(inspect.unwrap(original)):
        completions.create = _wrap_async(original, state)
    else:
        completions.create = _wrap_sync(original, state)
    completions._multimind_guarded = True
    return client


def _wrap_sync(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_request(kwargs, state)
        result = original(*args, **kwargs)
        if kwargs.get("stream"):
            return _GuardedStream(result, state, kwargs)
        return _screen_response(result, kwargs, state)

    return create


def _wrap_async(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_request(kwargs, state)
        result = await original(*args, **kwargs)
        if kwargs.get("stream"):
            return _GuardedAsyncStream(result, state, kwargs)
        return _screen_response(result, kwargs, state)

    return create


def _screen_request(kwargs: Dict[str, Any], state: GuardState) -> Dict[str, Any]:
    messages = kwargs.get("messages")
    if not isinstance(messages, (list, tuple)):
        return kwargs
    screened = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            screened.append({**m, "content": state.screen_input(m["content"], _METHOD)})
        elif isinstance(m, dict) and isinstance(m.get("content"), list):
            parts = [
                {**p, "text": state.screen_input(p["text"], _METHOD)}
                if isinstance(p, dict) and isinstance(p.get("text"), str)
                else p
                for p in m["content"]
            ]
            screened.append({**m, "content": parts})
        else:
            screened.append(m)
    return {**kwargs, "messages": screened}


def _input_text(kwargs: Dict[str, Any]) -> str:
    parts = []
    for m in kwargs.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            parts.append(m["content"])
    return "\n".join(parts)


def _screen_response(result: Any, kwargs: Dict[str, Any], state: GuardState) -> Any:
    output_parts = []
    for choice in getattr(result, "choices", None) or []:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            message.content = state.screen_output(content, _METHOD)
            output_parts.append(message.content)
    model = getattr(result, "model", None) or kwargs.get("model", "unknown")
    state.record_usage(
        str(model),
        _METHOD,
        input_text=_input_text(kwargs),
        output_text="\n".join(output_parts),
        usage=_usage_tokens(getattr(result, "usage", None)),
        provider="openai",
    )
    return result


class _StreamStateMixin:
    def __init__(self, stream: Any, state: GuardState, kwargs: Dict[str, Any]):
        self._stream = stream
        self._state = state
        self._model = str(kwargs.get("model", "unknown"))
        self._input = _input_text(kwargs)
        self._chunks: list = []
        self._done = False

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_stream"), name)

    def _collect(self, chunk: Any) -> None:
        for choice in getattr(chunk, "choices", None) or []:
            content = getattr(getattr(choice, "delta", None), "content", None)
            if isinstance(content, str):
                self._chunks.append(content)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        text = "".join(self._chunks)
        self._state.scan_output(text, _METHOD + ":stream")
        self._state.record_usage(
            self._model,
            _METHOD + ":stream",
            input_text=self._input,
            output_text=text,
            provider="openai",
        )


class _GuardedStream(_StreamStateMixin):
    """Pass-through wrapper for a sync openai Stream; post-scans after the last chunk."""

    def __iter__(self):
        for chunk in self._stream:
            self._collect(chunk)
            yield chunk
        self._finish()

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc: Any):
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if callable(exit_) else None


class _GuardedAsyncStream(_StreamStateMixin):
    """Pass-through wrapper for an async openai Stream; post-scans after the last chunk."""

    async def __aiter__(self):
        async for chunk in self._stream:
            self._collect(chunk)
            yield chunk
        self._finish()

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc: Any):
        exit_ = getattr(self._stream, "__aexit__", None)
        return await exit_(*exc) if callable(exit_) else None
