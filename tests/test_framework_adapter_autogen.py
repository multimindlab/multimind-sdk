"""Tests for the AutoGen/autogen-core adapter (multimind.integrations.frameworks.autogen).

autogen-core is a lightweight package (its only extra dependency is a
protobuf pin) so, unlike crewai/langchain-core/llama-index-core, it is
installed directly in this project's dev venv and exercised for real here
rather than via a sys.modules fake.
"""

import asyncio
import importlib
import io
import json
import sys

import pytest

from multimind.compliance.guard import ComplianceViolationError
from multimind.observability.cost_tracker import Budget, BudgetExceededError, CostTracker

autogen_core = pytest.importorskip("autogen_core")

from autogen_core.models import (  # noqa: E402
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)

from multimind.integrations.frameworks.autogen import (  # noqa: E402
    GuardedChatCompletionClient,
    guard_autogen_client,
)

EMAIL_TEXT = "Contact alice@example.com about the invoice."
PII_RESPONSE = "Reply to bob@x.io as soon as possible."
CLEAN_RESPONSE = "All good, nothing to report."


def audit_records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class FakeChatCompletionClient(ChatCompletionClient):
    """A from-scratch ChatCompletionClient implementing every abstract method,
    mirroring what a real OpenAIChatCompletionClient-style client looks like.
    """

    def __init__(self, response=CLEAN_RESPONSE, model="fake-model"):
        self.response = response
        self.model = model
        self.seen = []
        self.stream_events = None

    async def create(
        self,
        messages,
        *,
        tools=(),
        tool_choice="auto",
        json_output=None,
        extra_create_args={},
        cancellation_token=None,
    ):
        self.seen.append(list(messages))
        return CreateResult(
            finish_reason="stop",
            content=self.response,
            usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
            cached=False,
        )

    async def create_stream(
        self,
        messages,
        *,
        tools=(),
        tool_choice="auto",
        json_output=None,
        extra_create_args={},
        cancellation_token=None,
    ):
        self.seen.append(list(messages))
        for part in self.stream_events or [self.response]:
            yield part
        yield CreateResult(
            finish_reason="stop",
            content=self.response,
            usage=RequestUsage(prompt_tokens=3, completion_tokens=2),
            cached=False,
        )

    def actual_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self):
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, *, tools=()):
        return 42

    def remaining_tokens(self, messages, *, tools=()):
        return 100

    async def close(self):
        self.closed = True

    @property
    def model_info(self):
        return ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family=ModelFamily.UNKNOWN,
            structured_output=False,
        )

    @property
    def capabilities(self):
        return self.model_info


class TestGuardedChatCompletionClient:
    def test_isinstance_chat_completion_client(self):
        guarded = guard_autogen_client(FakeChatCompletionClient())
        assert isinstance(guarded, ChatCompletionClient)
        assert isinstance(guarded, GuardedChatCompletionClient)

    async def test_create_redacts_input_and_output(self):
        fake = FakeChatCompletionClient(response=PII_RESPONSE)
        guarded = guard_autogen_client(fake)
        result = await guarded.create(
            [SystemMessage(content="be nice"), UserMessage(content=EMAIL_TEXT, source="user")]
        )
        assert fake.seen[0][1].content == "Contact [EMAIL] about the invoice."
        assert "[EMAIL]" in result.content and "bob@x.io" not in result.content

    async def test_user_message_list_content_redacted(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake)
        await guarded.create([UserMessage(content=[EMAIL_TEXT, "clean"], source="user")])
        assert fake.seen[0][0].content == ["Contact [EMAIL] about the invoice.", "clean"]

    async def test_assistant_message_text_redacted(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake)
        await guarded.create([AssistantMessage(content=EMAIL_TEXT, source="assistant")])
        assert fake.seen[0][0].content == "Contact [EMAIL] about the invoice."

    async def test_function_execution_result_message_redacted(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake)
        fem = FunctionExecutionResultMessage(
            content=[
                FunctionExecutionResult(
                    content=EMAIL_TEXT, name="tool", call_id="1", is_error=False
                )
            ]
        )
        await guarded.create([fem])
        assert fake.seen[0][0].content[0].content == "Contact [EMAIL] about the invoice."

    async def test_records_exact_usage(self):
        tracker = CostTracker()
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake, tracker=tracker, pricing={"fake": 0.00001})
        await guarded.create([UserMessage(content="hi", source="user")])
        (record,) = tracker.records
        assert record.provider == "autogen" and record.model == "fake-model"
        assert (record.input_tokens, record.output_tokens) == (10, 5)
        assert record.estimated is False

    async def test_create_stream_passthrough_and_usage(self):
        tracker = CostTracker()
        stream = io.StringIO()
        fake = FakeChatCompletionClient(response=PII_RESPONSE)
        guarded = guard_autogen_client(fake, tracker=tracker, audit_log=stream)
        chunks = [
            item async for item in guarded.create_stream([UserMessage(content="hi", source="user")])
        ]
        # str chunks pass through unredacted; the terminal CreateResult is redacted
        assert chunks[0] == PII_RESPONSE
        assert isinstance(chunks[-1], CreateResult)
        assert "bob@x.io" not in chunks[-1].content
        (record,) = tracker.records
        assert record.method == "create_stream"
        assert (record.input_tokens, record.output_tokens) == (3, 2)
        output_events = [r for r in audit_records(stream) if r["direction"] == "output"]
        assert output_events[-1]["pii_types"] == {"email": 1}

    def test_block_on_raises(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake, block_on=("email",))

        async def run():
            await guarded.create([UserMessage(content=EMAIL_TEXT, source="user")])

        with pytest.raises(ComplianceViolationError):
            asyncio.run(run())

    async def test_budget_raises_before_next_call(self):
        budget = Budget(max_cost=0.0000001)
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(
            fake, tracker=CostTracker(), budget=budget, pricing={"fake": 0.00001}
        )
        await guarded.create([UserMessage(content="first", source="user")])
        with pytest.raises(BudgetExceededError):
            await guarded.create([UserMessage(content="second", source="user")])

    async def test_proxied_abstract_methods_delegate(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake)
        assert guarded.count_tokens([]) == 42
        assert guarded.remaining_tokens([]) == 100
        assert guarded.model_info["family"] == ModelFamily.UNKNOWN
        assert guarded.capabilities["family"] == ModelFamily.UNKNOWN
        await guarded.close()
        assert fake.closed is True

    def test_arbitrary_attrs_proxied(self):
        fake = FakeChatCompletionClient()
        guarded = guard_autogen_client(fake)
        assert guarded.response == CLEAN_RESPONSE  # not part of the ABC; proxied


class TestImportSafety:
    def test_import_error_when_autogen_core_absent(self, monkeypatch):
        for name in list(sys.modules):
            if name.startswith("autogen_core") or name.startswith(
                "multimind.integrations.frameworks.autogen"
            ):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "autogen_core", None)
        with pytest.raises(ImportError, match="autogen-core"):
            importlib.import_module("multimind.integrations.frameworks.autogen")
