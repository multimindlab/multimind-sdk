"""Governance adapter for raw OpenAI clients.

:func:`guard_openai` patches ``client.chat.completions.create`` on an
``openai.OpenAI`` or ``openai.AsyncOpenAI`` instance in place, so existing
call sites get PII redaction, budget enforcement, cost tracking, and audit
logging with no other change. Return types are preserved: the completion
object is returned as-is (with redacted message content), and streaming
returns a thin pass-through wrapper that scans/records after the last chunk.
``client.responses.create`` (the Responses API) and ``client.embeddings.create``
are wrapped the same way when present on the client — duck-typed and
absence-tolerant, so older SDK versions without one or the other still work.

Duck-typed on purpose — works against any client object whose
``chat.completions.create`` follows the OpenAI signature (Azure OpenAI,
compatible proxies, test fakes).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Dict, List

from multimind.observability.cost_tracker import _usage_tokens

from ._base import GuardState

__all__ = ["guard_openai"]

_METHOD = "chat.completions.create"
_METHOD_RESPONSES = "responses.create"
_METHOD_EMBEDDINGS = "embeddings.create"


def guard_openai(client: Any, **guard_kwargs: Any) -> Any:
    """Guard an OpenAI client in place: ``client = guard_openai(OpenAI(), budget=Budget(5.0))``.

    Accepts the shared governance kwargs (``redact_input``, ``redact_output``,
    ``strategy``, ``block_on``, ``detector``, ``audit_log``, ``tracker``,
    ``budget``, ``tag``, ``pricing``). Returns the same client instance.

    Wraps ``chat.completions.create`` (always), plus ``responses.create`` and
    ``embeddings.create`` when the client exposes them — each guarded
    independently, so a client missing one (older SDKs predate the Responses
    API) still gets the other two.
    """
    state = GuardState("openai", **guard_kwargs)
    completions = client.chat.completions
    if not getattr(completions, "_multimind_guarded", False):
        original = completions.create
        if inspect.iscoroutinefunction(inspect.unwrap(original)):
            completions.create = _wrap_async(original, state)
        else:
            completions.create = _wrap_sync(original, state)
        completions._multimind_guarded = True
    _guard_responses(client, state)
    _guard_embeddings(client, state)
    return client


def _guard_responses(client: Any, state: GuardState) -> None:
    responses = getattr(client, "responses", None)
    original = getattr(responses, "create", None)
    if responses is None or original is None or getattr(responses, "_multimind_guarded", False):
        return
    if inspect.iscoroutinefunction(inspect.unwrap(original)):
        responses.create = _wrap_responses_async(original, state)
    else:
        responses.create = _wrap_responses_sync(original, state)
    responses._multimind_guarded = True


def _guard_embeddings(client: Any, state: GuardState) -> None:
    embeddings = getattr(client, "embeddings", None)
    original = getattr(embeddings, "create", None)
    if embeddings is None or original is None or getattr(embeddings, "_multimind_guarded", False):
        return
    if inspect.iscoroutinefunction(inspect.unwrap(original)):
        embeddings.create = _wrap_embeddings_async(original, state)
    else:
        embeddings.create = _wrap_embeddings_sync(original, state)
    embeddings._multimind_guarded = True


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


def _screen_responses_request(kwargs: Dict[str, Any], state: GuardState) -> Dict[str, Any]:
    """Screen ``instructions`` and ``input`` for ``client.responses.create``.

    ``input`` is a string or a list of message-shaped dicts whose ``content``
    is a string or a list of content-part dicts (``{"type": "input_text",
    "text": ...}``); non-text parts (images, tool calls) pass through
    unscreened, matching the chat.completions handling of multimodal content.
    """
    kwargs = dict(kwargs)
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str):
        kwargs["instructions"] = state.screen_input(instructions, _METHOD_RESPONSES)
    input_ = kwargs.get("input")
    if isinstance(input_, str):
        kwargs["input"] = state.screen_input(input_, _METHOD_RESPONSES)
    elif isinstance(input_, (list, tuple)):
        kwargs["input"] = [_screen_responses_item(item, state) for item in input_]
    return kwargs


def _screen_responses_item(item: Any, state: GuardState) -> Any:
    if not isinstance(item, dict):
        return item
    content = item.get("content")
    if isinstance(content, str):
        return {**item, "content": state.screen_input(content, _METHOD_RESPONSES)}
    if isinstance(content, list):
        parts = [
            {**p, "text": state.screen_input(p["text"], _METHOD_RESPONSES)}
            if isinstance(p, dict) and isinstance(p.get("text"), str)
            else p
            for p in content
        ]
        return {**item, "content": parts}
    return item


def _responses_input_text(kwargs: Dict[str, Any]) -> str:
    parts: List[str] = []
    instructions = kwargs.get("instructions")
    if isinstance(instructions, str):
        parts.append(instructions)
    input_ = kwargs.get("input")
    if isinstance(input_, str):
        parts.append(input_)
    elif isinstance(input_, (list, tuple)):
        for item in input_:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
    return "\n".join(parts)


def _screen_responses_result(result: Any, kwargs: Dict[str, Any], state: GuardState) -> Any:
    output_parts = []
    for item in getattr(result, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                part.text = state.screen_output(text, _METHOD_RESPONSES)
                output_parts.append(part.text)
    model = getattr(result, "model", None) or kwargs.get("model", "unknown")
    state.record_usage(
        str(model),
        _METHOD_RESPONSES,
        input_text=_responses_input_text(kwargs),
        output_text="\n".join(output_parts),
        usage=_usage_tokens(getattr(result, "usage", None)),
        provider="openai",
    )
    return result


def _wrap_responses_sync(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_responses_request(kwargs, state)
        result = original(*args, **kwargs)
        if kwargs.get("stream"):
            return _GuardedResponsesStream(result, state, kwargs)
        return _screen_responses_result(result, kwargs, state)

    return create


def _wrap_responses_async(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_responses_request(kwargs, state)
        result = await original(*args, **kwargs)
        if kwargs.get("stream"):
            return _GuardedResponsesAsyncStream(result, state, kwargs)
        return _screen_responses_result(result, kwargs, state)

    return create


def _screen_embeddings_request(kwargs: Dict[str, Any], state: GuardState) -> Dict[str, Any]:
    """Screen ``input`` for ``client.embeddings.create``; token-id inputs pass through."""
    input_ = kwargs.get("input")
    if isinstance(input_, str):
        return {**kwargs, "input": state.screen_input(input_, _METHOD_EMBEDDINGS)}
    if isinstance(input_, (list, tuple)) and all(isinstance(i, str) for i in input_):
        return {**kwargs, "input": [state.screen_input(i, _METHOD_EMBEDDINGS) for i in input_]}
    return kwargs


def _embeddings_input_text(kwargs: Dict[str, Any]) -> str:
    input_ = kwargs.get("input")
    if isinstance(input_, str):
        return input_
    if isinstance(input_, (list, tuple)) and all(isinstance(i, str) for i in input_):
        return "\n".join(input_)
    return ""


def _record_embeddings_usage(result: Any, kwargs: Dict[str, Any], state: GuardState) -> Any:
    model = getattr(result, "model", None) or kwargs.get("model", "unknown")
    state.record_usage(
        str(model),
        _METHOD_EMBEDDINGS,
        input_text=_embeddings_input_text(kwargs),
        usage=_usage_tokens(getattr(result, "usage", None)),
        provider="openai",
    )
    return result


def _wrap_embeddings_sync(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_embeddings_request(kwargs, state)
        result = original(*args, **kwargs)
        return _record_embeddings_usage(result, kwargs, state)

    return create


def _wrap_embeddings_async(original: Any, state: GuardState) -> Any:
    @functools.wraps(original)
    async def create(*args: Any, **kwargs: Any) -> Any:
        state.check_budget()
        kwargs = _screen_embeddings_request(kwargs, state)
        result = await original(*args, **kwargs)
        return _record_embeddings_usage(result, kwargs, state)

    return create


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


class _ResponsesStreamStateMixin:
    """Like :class:`_StreamStateMixin` but for Responses API stream events,
    which carry deltas as ``event.delta`` (``type == "response.output_text.delta"``)
    and final usage on the terminal ``response.completed`` event's ``.response``.
    """

    def __init__(self, stream: Any, state: GuardState, kwargs: Dict[str, Any]):
        self._stream = stream
        self._state = state
        self._model = str(kwargs.get("model", "unknown"))
        self._input = _responses_input_text(kwargs)
        self._chunks: list = []
        self._usage: Any = None
        self._done = False

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_stream"), name)

    def _collect(self, event: Any) -> None:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str) and getattr(event, "type", "") == "response.output_text.delta":
            self._chunks.append(delta)
        response = getattr(event, "response", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage = usage

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        text = "".join(self._chunks)
        self._state.scan_output(text, _METHOD_RESPONSES + ":stream")
        self._state.record_usage(
            self._model,
            _METHOD_RESPONSES + ":stream",
            input_text=self._input,
            output_text=text,
            usage=_usage_tokens(self._usage),
            provider="openai",
        )


class _GuardedResponsesStream(_ResponsesStreamStateMixin):
    """Pass-through wrapper for a sync Responses API stream; post-scans after the last event."""

    def __iter__(self):
        for event in self._stream:
            self._collect(event)
            yield event
        self._finish()

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc: Any):
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if callable(exit_) else None


class _GuardedResponsesAsyncStream(_ResponsesStreamStateMixin):
    """Pass-through wrapper for an async Responses API stream; post-scans after the last event."""

    async def __aiter__(self):
        async for event in self._stream:
            self._collect(event)
            yield event
        self._finish()

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc: Any):
        exit_ = getattr(self._stream, "__aexit__", None)
        return await exit_(*exc) if callable(exit_) else None
