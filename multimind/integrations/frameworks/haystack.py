"""Governance adapter for Haystack 2.x chat generators.

:func:`guard_haystack_generator` wraps a Haystack 2.x chat generator
component (``OpenAIChatGenerator``, ``AzureOpenAIChatGenerator``, or any
component exposing the documented ``run(messages: List[ChatMessage], ...) ->
{"replies": List[ChatMessage]}`` contract, plus an optional async
``run_async``) with MultiMind's PII guard, audit trail, cost tracking, and
budget enforcement.

Deliberately framework-import-free: this module never imports ``haystack``.
Message screening is built against Haystack 2.x's *documented public*
``ChatMessage`` API (``.text``, ``.role``, ``.meta``, and the
``ChatMessage.from_system``/``from_user``/``from_assistant`` factory
methods) rather than its private dataclass fields, and is applied via
``getattr``/``hasattr`` duck-typing — so no import is needed to redact a
real ``ChatMessage``, and the adapter never breaks merely because
``haystack-ai`` (a heavy dependency) is absent. Because the project's policy
is not to install ``haystack-ai`` just to exercise this adapter, it is
verified here against a ``sys.modules`` fake mirroring that documented
shape, not the installed package — report interface drift as a bug.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ._base import GuardState

__all__ = ["GuardedHaystackGenerator", "guard_haystack_generator"]

_METHOD_RUN = "run"
_METHOD_RUN_ASYNC = "run_async"


class GuardedHaystackGenerator:
    """PII/cost/budget guard around a Haystack chat generator component.

    Wraps ``run`` (always) and ``run_async`` (if the wrapped component
    defines one); every other attribute is proxied to the wrapped component.

    Not itself a ``@component``-decorated Haystack component (building the
    dynamic input/output sockets Haystack's decorator attaches at class
    definition time isn't something a runtime wrapper can do safely without
    the real package to verify against), so it cannot be inserted into a
    ``Pipeline`` via ``add_component`` — call ``.run()``/``.run_async()``
    directly, or guard a generator used outside pipeline orchestration.
    """

    def __init__(self, component: Any, state: GuardState):
        self._component = component
        self._state = state

    def __getattr__(self, name: str) -> Any:
        try:
            component = object.__getattribute__(self, "_component")
        except AttributeError:
            raise AttributeError(name)
        return getattr(component, name)

    def run(self, messages: List[Any], **kwargs: Any) -> Dict[str, Any]:
        self._state.check_budget()
        screened = [self._screen_message(m, _METHOD_RUN) for m in messages]
        result = self._component.run(screened, **kwargs)
        return self._screen_result(screened, result, _METHOD_RUN)

    async def run_async(self, messages: List[Any], **kwargs: Any) -> Dict[str, Any]:
        run_async = getattr(self._component, "run_async", None)
        if run_async is None:
            raise AttributeError(f"{type(self._component).__name__!r} has no run_async method")
        self._state.check_budget()
        screened = [self._screen_message(m, _METHOD_RUN_ASYNC) for m in messages]
        result = await run_async(screened, **kwargs)
        return self._screen_result(screened, result, _METHOD_RUN_ASYNC)

    # -- helpers ------------------------------------------------------------

    def _screen_message(self, message: Any, method: str) -> Any:
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            return message  # tool calls / non-text content: pass through
        return self._rebuild(message, self._state.screen_input(text, method))

    def _screen_reply(self, message: Any, method: str) -> Any:
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            return message
        return self._rebuild(message, self._state.screen_output(text, method))

    @staticmethod
    def _rebuild(message: Any, text: str) -> Any:
        if text == getattr(message, "text", None):
            return message
        role = getattr(message, "role", None)
        role_value = getattr(role, "value", role)
        meta = dict(getattr(message, "meta", None) or {})
        cls = type(message)
        if role_value == "system" and hasattr(cls, "from_system"):
            return cls.from_system(text, meta=meta)
        if role_value == "user" and hasattr(cls, "from_user"):
            return cls.from_user(text, meta=meta)
        if role_value == "assistant" and hasattr(cls, "from_assistant"):
            return cls.from_assistant(text, meta=meta, name=getattr(message, "name", None))
        # Tool-role messages (and any unrecognized role) need data this
        # helper doesn't have (the originating ToolCall / tool_call_result)
        # to reconstruct safely — leave unmodified rather than guess.
        return message

    def _screen_result(self, requests: List[Any], result: Any, method: str) -> Any:
        if not isinstance(result, dict) or not isinstance(result.get("replies"), list):
            return result
        replies = [self._screen_reply(r, method) for r in result["replies"]]
        input_text = "\n".join(
            getattr(m, "text", "") for m in requests if isinstance(getattr(m, "text", None), str)
        )
        output_text = "\n".join(
            getattr(r, "text", "") for r in replies if isinstance(getattr(r, "text", None), str)
        )
        model, usage = self._usage_of(replies)
        self._state.record_usage(
            model,
            method,
            input_text=input_text,
            output_text=output_text,
            usage=usage,
            provider="haystack",
        )
        return {**result, "replies": replies}

    @staticmethod
    def _usage_of(replies: List[Any]) -> Tuple[str, Optional[Tuple[int, int]]]:
        for reply in replies:
            meta = getattr(reply, "meta", None) or {}
            if not isinstance(meta, dict):
                continue
            model = str(meta.get("model", "unknown"))
            usage = meta.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                if input_tokens is not None or output_tokens is not None:
                    return model, (input_tokens or 0, output_tokens or 0)
        return "unknown", None


def guard_haystack_generator(component: Any, **guard_kwargs: Any) -> GuardedHaystackGenerator:
    """Wrap a Haystack chat generator: ``guard_haystack_generator(OpenAIChatGenerator())``.

    Accepts the shared governance kwargs (``redact_input``, ``redact_output``,
    ``strategy``, ``block_on``, ``detector``, ``audit_log``, ``tracker``,
    ``budget``, ``tag``, ``pricing``).
    """
    return GuardedHaystackGenerator(component, GuardState("haystack", **guard_kwargs))
