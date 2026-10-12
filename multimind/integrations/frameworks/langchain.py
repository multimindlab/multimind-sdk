"""Governance adapters for LangChain.

Entry points, all additive to an existing LangChain app:

* :func:`guard_runnable` — PII guard around any Runnable/chat model.
* :func:`guard_tool` / :func:`guard_tools` — PII guard around agent tools, so
  tool *inputs* (not just LLM prompts) are screened before the tool body runs.
* :class:`MultiMindChatModel` — a ``BaseChatModel`` backed by any MultiMind
  BaseLLM-compatible model, so MultiMind providers slot into LCEL chains.
* :class:`MultiMindCallbackHandler` — cost/audit/budget events from any
  LangChain run via ``callbacks=[handler]``; zero wrapping needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple
from uuid import UUID

_INSTALL_HINT = (
    "multimind.integrations.frameworks.langchain requires the `langchain-core` "
    "package. Install with: pip install langchain-core"
)

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
    from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - exercised via import-safety test
    raise ImportError(_INSTALL_HINT) from exc

from multimind.observability.cost_tracker import _usage_tokens, get_default_tracker

from ._base import GuardState

__all__ = [
    "GuardedRunnable",
    "GuardedTool",
    "MultiMindCallbackHandler",
    "MultiMindChatModel",
    "guard_runnable",
    "guard_tool",
    "guard_tools",
]

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _run_coro(coro: Any, method: str) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        f"{method}() cannot be called from a running event loop; use the async variant instead."
    )


class GuardedRunnable(Runnable):
    """PII guard around any LangChain Runnable; a Runnable itself, so it drops
    into LCEL chains, agents, and ``|`` composition unchanged.

    String content in inputs (str, messages, prompt values, dict values) is
    screened/redacted before it reaches the wrapped runnable; ``invoke`` /
    ``ainvoke`` outputs are scanned and redacted; ``stream`` / ``astream``
    pass chunks through untouched and scan the accumulated text afterwards.
    Non-string content (multimodal blocks, tool call arguments) is passed
    through unscreened.
    """

    def __init__(self, bound: Runnable, state: GuardState):
        self.bound = bound
        self._state = state

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        input = self._screen_in(input, "invoke")
        return self._screen_out(self.bound.invoke(input, config=config, **kwargs), "invoke")

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        input = self._screen_in(input, "ainvoke")
        output = await self.bound.ainvoke(input, config=config, **kwargs)
        return self._screen_out(output, "ainvoke")

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Iterator[Any]:
        input = self._screen_in(input, "stream")
        collected: List[str] = []
        for chunk in self.bound.stream(input, config=config, **kwargs):
            collected.append(self._chunk_text(chunk))
            yield chunk
        self._state.scan_output("".join(collected), "stream")

    async def astream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        input = self._screen_in(input, "astream")
        collected: List[str] = []
        async for chunk in self.bound.astream(input, config=config, **kwargs):
            collected.append(self._chunk_text(chunk))
            yield chunk
        self._state.scan_output("".join(collected), "astream")

    def _screen_in(self, value: Any, method: str) -> Any:
        if isinstance(value, str):
            return self._state.screen_input(value, method)
        if isinstance(value, BaseMessage):
            return self._screen_message(value, method)
        if isinstance(value, StringPromptValue):
            return StringPromptValue(text=self._state.screen_input(value.text, method))
        if isinstance(value, ChatPromptValue):
            return ChatPromptValue(
                messages=[self._screen_message(m, method) for m in value.to_messages()]
            )
        if isinstance(value, PromptValue):
            return [self._screen_message(m, method) for m in value.to_messages()]
        if isinstance(value, dict):
            return {
                k: self._state.screen_input(v, method) if isinstance(v, str) else v
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(self._screen_in(v, method) for v in value)
        return value

    def _screen_message(self, message: BaseMessage, method: str) -> BaseMessage:
        if isinstance(message.content, str):
            return message.model_copy(
                update={"content": self._state.screen_input(message.content, method)}
            )
        return message

    def _screen_out(self, value: Any, method: str) -> Any:
        if isinstance(value, str):
            return self._state.screen_output(value, method)
        if isinstance(value, BaseMessage) and isinstance(value.content, str):
            return value.model_copy(
                update={"content": self._state.screen_output(value.content, method)}
            )
        return value

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        content = getattr(chunk, "content", None)
        return content if isinstance(content, str) else ""


def guard_runnable(runnable: Runnable, **guard_kwargs: Any) -> GuardedRunnable:
    """Wrap any Runnable with MultiMind's PII guard: ``guard_runnable(chain, block_on=("ssn",))``.

    Accepts the same governance kwargs as :class:`ComplianceGuard`
    (``redact_input``, ``redact_output``, ``strategy``, ``block_on``,
    ``detector``, ``audit_log``).
    """
    return GuardedRunnable(runnable, GuardState("langchain", **guard_kwargs))


class GuardedTool(BaseTool):
    """PII guard around a LangChain ``Tool``/``BaseTool``; a ``BaseTool``
    itself, so it drops into an agent's tool list unchanged and keeps the
    wrapped tool's ``name``/``description``/``args_schema`` for LLM binding.

    Only the tool's *input* — the string or dict argument an agent passes to
    it — is screened/redacted before it reaches the wrapped tool's
    ``_run``/``_arun``; the tool's return value is not screened here (route
    the agent's LLM through :func:`guard_runnable` to catch PII flowing back
    from tool output through the model).
    """

    def __init__(self, tool: BaseTool, state: GuardState, **kwargs: Any):
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
            **kwargs,
        )
        object.__setattr__(self, "_tool", tool)
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name: str) -> Any:
        try:
            tool = object.__getattribute__(self, "_tool")
        except AttributeError:
            raise AttributeError(name)
        return getattr(tool, name)

    def _run(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> Any:
        tool_input = self._screen(args, kwargs, "invoke")
        return self._tool.run(tool_input)

    async def _arun(self, *args: Any, run_manager: Any = None, **kwargs: Any) -> Any:
        tool_input = self._screen(args, kwargs, "ainvoke")
        return await self._tool.arun(tool_input)

    def _screen(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], method: str) -> Any:
        # Mirrors BaseTool._to_args_and_kwargs: a single positional string for
        # string-input tools, or an all-keyword dict for structured tools.
        if args:
            value = args[0]
            return self._state.screen_input(value, method) if isinstance(value, str) else value
        return {
            k: self._state.screen_input(v, method) if isinstance(v, str) else v
            for k, v in kwargs.items()
        }


def guard_tool(tool: BaseTool, **guard_kwargs: Any) -> GuardedTool:
    """Wrap one LangChain tool so its input is screened before the tool runs.

    ``Agent(tools=[guard_tool(search_tool, block_on=("ssn",))])``. Accepts the
    same governance kwargs as :func:`guard_runnable`.
    """
    return GuardedTool(tool, GuardState("langchain", **guard_kwargs))


def guard_tools(tools: List[BaseTool], **guard_kwargs: Any) -> List[BaseTool]:
    """Wrap every tool in a list: ``guard_tools([search_tool, calc_tool])``."""
    return [guard_tool(t, **guard_kwargs) for t in tools]


class MultiMindChatModel(BaseChatModel):
    """LangChain chat model backed by any MultiMind BaseLLM-compatible model.

    ``MultiMindChatModel(model=ClaudeModel(...))`` behaves like any other
    LangChain chat model, so MultiMind providers (and MultiMind guards
    wrapped around them) participate in LCEL chains natively.
    """

    model: Any

    @property
    def _llm_type(self) -> str:
        return "multimind"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": getattr(self.model, "model_name", "unknown")}

    @staticmethod
    def _to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        out = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            out.append({"role": _ROLE_MAP.get(m.type, m.type), "content": content})
        return out

    @staticmethod
    def _to_result(text: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(text)))])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop is not None:
            kwargs["stop"] = stop
        chat_sync = getattr(self.model, "chat_sync", None)
        if callable(chat_sync):
            text = chat_sync(self._to_dicts(messages), **kwargs)
        else:
            text = _run_coro(self.model.chat(self._to_dicts(messages), **kwargs), "invoke")
        return self._to_result(text)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if stop is not None:
            kwargs["stop"] = stop
        text = await self.model.chat(self._to_dicts(messages), **kwargs)
        return self._to_result(text)


class MultiMindCallbackHandler(BaseCallbackHandler):
    """Streams cost/audit events from any LangChain run into MultiMind.

    Attach with ``callbacks=[handler]`` (per-call config or constructor) —
    nothing needs wrapping. Records every LLM call into a
    :class:`CostTracker` (real token usage when the provider reports it,
    chars/4 estimate otherwise), enforces an optional :class:`Budget`
    (raising ``BudgetExceededError`` before the next call once exceeded),
    and writes lifecycle events to an optional :class:`AuditLog`.

    This handler is observe-only by construction, not by omission: LangChain's
    ``CallbackManager`` invokes every handler's ``on_llm_start``/
    ``on_chat_model_start`` purely for its side effects and discards the
    return value (see ``langchain_core.callbacks.manager.handle_event``), so
    no callback — this one included — can rewrite the prompt actually sent to
    the model. Since redaction is off the table here, ``on_llm_start`` instead
    runs the same PII detector used by the guard wrappers over the outgoing
    prompt and records a ``pii_detected`` audit event (types/counts only,
    never raw text) *before* the call leaves the process — use
    :func:`guard_runnable` alongside this handler when redaction is required.
    """

    raise_error = True  # budget violations must propagate, not be logged away

    def __init__(
        self,
        tracker: Any = None,
        budget: Any = None,
        audit_log: Any = None,
        tag: Optional[str] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ):
        self._state = GuardState(
            "langchain",
            tracker=tracker if tracker is not None else get_default_tracker(),
            budget=budget,
            audit_log=audit_log,
            tag=tag,
            pricing=pricing,
        )
        self._runs: Dict[UUID, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        text = "\n".join(
            m.content for batch in messages for m in batch if isinstance(m.content, str)
        )
        self._start(serialized, text, run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self._start(serialized, "\n".join(prompts), run_id, metadata, kwargs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        run = self._runs.pop(run_id, {})
        output_text = "".join(g.text for gens in response.generations for g in gens)
        info = self._state.record_usage(
            run.get("model", "unknown"),
            "llm",
            input_text=run.get("input", ""),
            output_text=output_text,
            usage=self._usage(response),
        )
        self._state.audit(
            {
                "event": "llm_end",
                "model": run.get("model", "unknown"),
                "run_id": str(run_id),
                **(info or {}),
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> Any:
        run = self._runs.pop(run_id, {})
        self._state.audit(
            {
                "event": "llm_error",
                "model": run.get("model", "unknown"),
                "run_id": str(run_id),
                "error": type(error).__name__,
            }
        )

    def _start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_text: str,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> None:
        self._state.check_budget()
        model = _model_name(serialized, metadata, kwargs)
        self._runs[run_id] = {"input": input_text, "model": model}
        self._state.audit({"event": "llm_start", "model": model, "run_id": str(run_id)})
        pii_types = self._state.detect_counts(input_text)
        if pii_types:
            self._state.audit(
                {
                    "event": "pii_detected",
                    "model": model,
                    "run_id": str(run_id),
                    "pii_types": pii_types,
                    "count": sum(pii_types.values()),
                }
            )

    @staticmethod
    def _usage(response: LLMResult) -> Optional[Tuple[int, int]]:
        llm_output = response.llm_output or {}
        usage = _usage_tokens(llm_output.get("token_usage") or llm_output.get("usage"))
        if usage is not None:
            return usage
        for gens in response.generations:
            for g in gens:
                usage = _usage_tokens(getattr(getattr(g, "message", None), "usage_metadata", None))
                if usage is not None:
                    return usage
        return None


def _model_name(
    serialized: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> str:
    metadata = metadata or {}
    serialized = serialized or {}
    params = kwargs.get("invocation_params") or {}
    ser_kwargs = serialized.get("kwargs") or {}
    for candidate in (
        metadata.get("ls_model_name"),
        params.get("model"),
        params.get("model_name"),
        ser_kwargs.get("model"),
        ser_kwargs.get("model_name"),
        serialized.get("name"),
    ):
        if candidate:
            return str(candidate)
    return "unknown"
