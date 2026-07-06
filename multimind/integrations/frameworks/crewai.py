"""Governance adapter for CrewAI.

:func:`guard_crew_llm` wraps a ``crewai.LLM`` (or any object exposing
CrewAI's ``call``/``acall`` interface) with MultiMind's PII guard, audit
trail, cost tracking, and budget enforcement. Pass the wrapper anywhere
CrewAI accepts an LLM instance (``Agent(llm=...)``, ``Crew(...)``).

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
    """

    def __init__(self, llm: Any, state: GuardState):
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_state", state)
        try:
            _CrewBaseLLM.__init__(
                self,
                model=str(getattr(llm, "model", "unknown")),
                temperature=getattr(llm, "temperature", None),
            )
        except TypeError:
            pass
        if hasattr(llm, "stop"):
            self.stop = llm.stop

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
            else:
                screened.append(m)
        return screened

    def _finish(self, screened: Any, result: Any, method: str) -> Any:
        if isinstance(screened, str):
            input_text = screened
        else:
            input_text = "\n".join(m.get("content", "") for m in screened if isinstance(m, dict))
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
