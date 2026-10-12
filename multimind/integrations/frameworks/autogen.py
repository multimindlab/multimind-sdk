"""Governance adapter for AutoGen / autogen-agentchat.

:func:`guard_autogen_client` wraps an ``autogen_core.models.ChatCompletionClient``
— the model-client interface AutoGen and autogen-agentchat agents call
through — with MultiMind's PII guard, audit trail, cost tracking, and budget
enforcement. Pass the wrapper anywhere an agent accepts a model client
(``AssistantAgent(model_client=...)``).

Interface choice: ``autogen_core.models.ChatCompletionClient``. It is the
documented, versioned abstract base class every AutoGen/autogen-agentchat
model client implements (``OpenAIChatCompletionClient``,
``AzureOpenAIChatCompletionClient``, and any custom subclass), and it is the
seam the AutoGen docs point to for building a custom model client. AG2's
alternative — a loosely-typed ``llm_config`` dict consumed by
``autogen.OpenAIWrapper`` — has no formal ABC and no stable method contract
to guard against, so ``ChatCompletionClient`` is the clearer, more stable
choice between the two.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Sequence, Union

_INSTALL_HINT = (
    "multimind.integrations.frameworks.autogen requires the `autogen-core` "
    "package. Install with: pip install autogen-core"
)

try:
    from autogen_core.models import (
        AssistantMessage,
        ChatCompletionClient,
        CreateResult,
        FunctionExecutionResultMessage,
        LLMMessage,
        SystemMessage,
        UserMessage,
    )
except ImportError as exc:  # pragma: no cover - exercised via import-safety test
    raise ImportError(_INSTALL_HINT) from exc

from ._base import GuardState

__all__ = ["GuardedChatCompletionClient", "guard_autogen_client"]

_METHOD_CREATE = "create"
_METHOD_STREAM = "create_stream"


class GuardedChatCompletionClient(ChatCompletionClient):
    """PII/cost/budget guard around an AutoGen ``ChatCompletionClient``.

    Wraps any concrete client (``OpenAIChatCompletionClient``, a custom
    subclass, a test fake, ...) so an agent built against the
    ``ChatCompletionClient`` interface gets redaction, budget enforcement,
    cost tracking, and audit logging with no other change. ``create`` and
    ``create_stream`` are guarded; every other abstract method
    (``actual_usage``, ``total_usage``, ``count_tokens``,
    ``remaining_tokens``, ``close``, ``model_info``, ``capabilities``) is
    proxied straight through to the wrapped client.

    Message content is screened per ``LLMMessage`` variant:
    ``SystemMessage.content`` (str) and ``UserMessage.content`` (str or list
    of str/``Image`` parts — only the str parts are screened) are user- and
    developer-authored, so both are redacted. ``AssistantMessage.content``
    is redacted when it is plain text; when it is a list of ``FunctionCall``
    (the model's own tool-call arguments) it passes through unscreened, same
    as the other adapters' tool-call handling.
    ``FunctionExecutionResultMessage`` (tool results fed back to the model)
    has its ``FunctionExecutionResult.content`` strings redacted too, since
    that is exactly the kind of external, unvetted text this guard exists
    to catch.
    """

    def __init__(self, client: ChatCompletionClient, state: GuardState):
        self._client = client
        self._state = state

    def __getattr__(self, name: str) -> Any:
        try:
            client = object.__getattribute__(self, "_client")
        except AttributeError:
            raise AttributeError(name)
        return getattr(client, name)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Dict[str, Any] = {},  # noqa: B006 - mirrors base signature
        cancellation_token: Any = None,
    ) -> CreateResult:
        self._state.check_budget()
        screened = [self._screen_message(m, _METHOD_CREATE) for m in messages]
        result = await self._client.create(
            screened,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        result = self._screen_result(result, _METHOD_CREATE)
        self._record(screened, result, _METHOD_CREATE)
        return result

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Dict[str, Any] = {},  # noqa: B006 - mirrors base signature
        cancellation_token: Any = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        self._state.check_budget()
        screened = [self._screen_message(m, _METHOD_STREAM) for m in messages]
        chunks: List[str] = []
        async for item in self._client.create_stream(
            screened,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        ):
            if isinstance(item, str):
                chunks.append(item)
                yield item
            else:
                # Terminal CreateResult: chunks already left the process
                # unredacted, so audit what was streamed and redact the
                # structured result callers actually hold onto.
                self._state.scan_output("".join(chunks), _METHOD_STREAM)
                result = self._screen_result(item, _METHOD_STREAM)
                self._record(screened, result, _METHOD_STREAM)
                yield result

    # -- proxied abstract methods -----------------------------------------

    def actual_usage(self) -> Any:
        return self._client.actual_usage()

    def total_usage(self) -> Any:
        return self._client.total_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return self._client.count_tokens(messages, tools=tools)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return self._client.remaining_tokens(messages, tools=tools)

    async def close(self) -> None:
        await self._client.close()

    @property
    def model_info(self) -> Any:
        return self._client.model_info

    @property
    def capabilities(self) -> Any:
        return self._client.capabilities

    # -- helpers ------------------------------------------------------------

    def _screen_message(self, message: LLMMessage, method: str) -> LLMMessage:
        if isinstance(message, SystemMessage):
            return message.model_copy(
                update={"content": self._state.screen_input(message.content, method)}
            )
        if isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, str):
                return message.model_copy(
                    update={"content": self._state.screen_input(content, method)}
                )
            if isinstance(content, list):
                screened = [
                    self._state.screen_input(part, method) if isinstance(part, str) else part
                    for part in content
                ]
                return message.model_copy(update={"content": screened})
            return message
        if isinstance(message, AssistantMessage):
            if isinstance(message.content, str):
                return message.model_copy(
                    update={"content": self._state.screen_input(message.content, method)}
                )
            return message  # list[FunctionCall]: the model's own tool-call args
        if isinstance(message, FunctionExecutionResultMessage):
            screened_results = [
                r.model_copy(update={"content": self._state.screen_input(r.content, method)})
                if isinstance(getattr(r, "content", None), str)
                else r
                for r in message.content
            ]
            return message.model_copy(update={"content": screened_results})
        return message

    def _screen_result(self, result: CreateResult, method: str) -> CreateResult:
        if isinstance(result.content, str):
            return result.model_copy(
                update={"content": self._state.screen_output(result.content, method)}
            )
        return result  # list[FunctionCall]: tool calls the model wants to make

    def _record(self, messages: List[LLMMessage], result: CreateResult, method: str) -> None:
        input_text = "\n".join(self._text_of(m) for m in messages)
        output_text = result.content if isinstance(result.content, str) else ""
        usage = result.usage
        self._state.record_usage(
            self._model_name(),
            method,
            input_text=input_text,
            output_text=output_text,
            usage=(usage.prompt_tokens, usage.completion_tokens) if usage is not None else None,
            provider="autogen",
        )

    def _model_name(self) -> str:
        model = getattr(self._client, "model", None)
        if model:
            return str(model)
        try:
            info = self._client.model_info
        except Exception:
            return "unknown"
        return str(info.get("family", "unknown")) if isinstance(info, dict) else "unknown"

    @staticmethod
    def _text_of(message: LLMMessage) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(p for p in content if isinstance(p, str))
        return ""


def guard_autogen_client(
    client: ChatCompletionClient, **guard_kwargs: Any
) -> GuardedChatCompletionClient:
    """Wrap an AutoGen model client: ``AssistantAgent(model_client=guard_autogen_client(client))``.

    Accepts the shared governance kwargs (``redact_input``, ``redact_output``,
    ``strategy``, ``block_on``, ``detector``, ``audit_log``, ``tracker``,
    ``budget``, ``tag``, ``pricing``).
    """
    return GuardedChatCompletionClient(client, GuardState("autogen", **guard_kwargs))
