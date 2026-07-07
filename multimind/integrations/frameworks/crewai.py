"""Governance adapter for CrewAI.

:func:`guard_crew_llm` wraps a ``crewai.LLM`` (or any object exposing
CrewAI's ``call``/``acall`` interface) with MultiMind's PII guard, audit
trail, cost tracking, and budget enforcement. Pass the wrapper anywhere
CrewAI accepts an LLM instance (``Agent(llm=...)``, ``Crew(...)``).

Verified against crewai 1.15.1 (installed and exercised directly, not just
faked): ``crewai.BaseLLM`` is a Pydantic v2 model whose ``call``/``acall``
take ``messages: str | list[LLMMessage]`` (``LLMMessage.content`` may be a
plain string or a list of content parts) and whose ``__init__`` rebuilds the
instance ``__dict__`` — see :class:`GuardedCrewLLM` for how that shapes
initialization order.

Budget enforcement is the headline here: agent crews are prone to runaway
token consumption (delegation loops, retries, verbose tool chatter). With a
``Budget`` attached, the guard raises ``BudgetExceededError`` before the
next LLM call once the ceiling is hit — stopping a crew mid-run instead of
letting it burn through spend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

_INSTALL_HINT = (
    "multimind.integrations.frameworks.crewai requires the `crewai` package. "
    "Install with: pip install crewai"
)

try:
    try:
        from crewai import BaseLLM as _CrewBaseLLM
    except ImportError:
        from crewai.llms.base_llm import BaseLLM as _CrewBaseLLM
except ImportError as exc:  # pragma: no cover - exercised via import-safety test
    raise ImportError(_INSTALL_HINT) from exc

from ._base import GuardState

__all__ = ["GuardedCrewLLM", "guard_crew_llm"]


class GuardedCrewLLM(_CrewBaseLLM):
    """PII/cost/budget guard around a CrewAI LLM.

    Subclasses ``crewai.BaseLLM`` so CrewAI's ``isinstance`` checks accept it
    as a first-class LLM. ``call``/``acall`` are guarded; every other
    attribute is proxied to the wrapped LLM.

    Verified against crewai 1.15.1, where ``BaseLLM`` is a Pydantic model
    whose ``__init__`` resets the instance ``__dict__`` — so the private
    guard references are attached *after* base initialization.
    """

    def __init__(self, llm: Any, state: GuardState):
        try:
            _CrewBaseLLM.__init__(
                self,
                model=str(getattr(llm, "model", "unknown")),
                temperature=getattr(llm, "temperature", None),
            )
        except TypeError:
            pass
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_state", state)
        stop = getattr(llm, "stop", None)
        if stop is not None:
            try:
                self.stop = stop
            except (TypeError, ValueError):  # pydantic field validation on 1.x
                pass

    def __getattr__(self, name: str) -> Any:
        try:
            llm = object.__getattribute__(self, "_llm")
        except AttributeError:
            raise AttributeError(name)
        return getattr(llm, name)

    def call(self, messages: Union[str, List[Dict[str, str]]], *args: Any, **kwargs: Any) -> Any:
        self._state.check_budget()
        screened = self._screen_messages(messages, "call")
        result = self._llm.call(screened, *args, **kwargs)
        return self._finish(screened, result, "call")

    async def acall(
        self, messages: Union[str, List[Dict[str, str]]], *args: Any, **kwargs: Any
    ) -> Any:
        self._state.check_budget()
        screened = self._screen_messages(messages, "acall")
        result = await self._llm.acall(screened, *args, **kwargs)
        return self._finish(screened, result, "acall")

    def supports_function_calling(self) -> bool:
        fn = getattr(self._llm, "supports_function_calling", None)
        return bool(fn()) if callable(fn) else False

    def supports_stop_words(self) -> bool:
        fn = getattr(self._llm, "supports_stop_words", None)
        return bool(fn()) if callable(fn) else False

    def supports_multimodal(self) -> bool:
        fn = getattr(self._llm, "supports_multimodal", None)
        return bool(fn()) if callable(fn) else False

    def get_context_window_size(self) -> int:
        fn = getattr(self._llm, "get_context_window_size", None)
        return int(fn()) if callable(fn) else 4096

    def _screen_messages(
        self, messages: Union[str, List[Dict[str, str]]], method: str
    ) -> Union[str, List[Dict[str, str]]]:
        if isinstance(messages, str):
            return self._state.screen_input(messages, method)
        screened = []
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                screened.append({**m, "content": self._state.screen_input(m["content"], method)})
            elif isinstance(m, dict) and isinstance(m.get("content"), list):
                # crewai 1.x LLMMessage content may be a list of parts
                parts = [
                    {**p, "text": self._state.screen_input(p["text"], method)}
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                    else p
                    for p in m["content"]
                ]
                screened.append({**m, "content": parts})
            else:
                screened.append(m)
        return screened

    def _finish(self, screened: Any, result: Any, method: str) -> Any:
        if isinstance(screened, str):
            input_text = screened
        else:
            input_text = "\n".join(
                m["content"]
                for m in screened
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            )
        output_text = ""
        if isinstance(result, str):
            result = self._state.screen_output(result, method)
            output_text = result
        self._state.record_usage(
            str(getattr(self._llm, "model", "unknown")),
            method,
            input_text=input_text,
            output_text=output_text,
        )
        return result


def guard_crew_llm(llm: Any, **guard_kwargs: Any) -> GuardedCrewLLM:
    """Wrap a CrewAI LLM: ``Agent(llm=guard_crew_llm(llm, budget=Budget(5.0)))``.

    Accepts the shared governance kwargs (``redact_input``, ``redact_output``,
    ``strategy``, ``block_on``, ``detector``, ``audit_log``, ``tracker``,
    ``budget``, ``tag``, ``pricing``).
    """
    return GuardedCrewLLM(llm, GuardState("crewai", **guard_kwargs))
