"""Tests for the Haystack adapter (multimind.integrations.frameworks.haystack).

haystack-ai is a heavy dependency (torch-adjacent extras) that the project
policy says not to install just to exercise one adapter. The adapter itself
is deliberately framework-import-free (it never imports `haystack`), so
these tests exercise it against a from-scratch fake mirroring Haystack
2.x's documented public `ChatMessage` API (`.text`, `.role`, `.meta`, and
the `ChatMessage.from_system`/`from_user`/`from_assistant` factories) rather
than a real installation or a sys.modules stub of the real package.
"""

import asyncio
from enum import Enum

import pytest

from multimind.compliance.guard import ComplianceViolationError
from multimind.integrations.frameworks.haystack import (
    GuardedHaystackGenerator,
    guard_haystack_generator,
)
from multimind.observability.cost_tracker import Budget, BudgetExceededError, CostTracker

EMAIL_TEXT = "Contact alice@example.com about the invoice."
PII_RESPONSE = "Reply to bob@x.io as soon as possible."
CLEAN_RESPONSE = "All good, nothing to report."


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FakeChatMessage:
    """Mirrors Haystack 2.x's public ChatMessage surface: `.text`, `.role`,
    `.meta`, `.name`, and the `from_system`/`from_user`/`from_assistant`
    factory classmethods.
    """

    def __init__(self, role, text, meta=None, name=None):
        self.role = role
        self.text = text
        self.meta = meta or {}
        self.name = name

    @classmethod
    def from_system(cls, text, meta=None):
        return cls(ChatRole.SYSTEM, text, meta)

    @classmethod
    def from_user(cls, text, meta=None):
        return cls(ChatRole.USER, text, meta)

    @classmethod
    def from_assistant(cls, text=None, meta=None, name=None):
        return cls(ChatRole.ASSISTANT, text, meta, name)


class FakeToolCallContent:
    """Non-text content (e.g. a tool call): has no `.text`, passes through."""

    def __init__(self):
        self.role = ChatRole.ASSISTANT


class FakeGenerator:
    def __init__(self, response=CLEAN_RESPONSE, model="fake-generator"):
        self.response = response
        self.model = model
        self.seen = []

    def run(self, messages, **kwargs):
        self.seen.append(messages)
        return {
            "replies": [
                FakeChatMessage.from_assistant(
                    self.response,
                    meta={
                        "model": self.model,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                )
            ]
        }

    async def run_async(self, messages, **kwargs):
        self.seen.append(messages)
        return {
            "replies": [
                FakeChatMessage.from_assistant(
                    self.response,
                    meta={
                        "model": self.model,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                )
            ]
        }


class SyncOnlyGenerator:
    """No run_async — the common case for a component that hasn't opted into
    Haystack's async support."""

    def run(self, messages, **kwargs):
        return {"replies": [FakeChatMessage.from_assistant(CLEAN_RESPONSE)]}


class TestGuardedHaystackGenerator:
    def test_run_redacts_input_and_output(self):
        gen = FakeGenerator(response=PII_RESPONSE)
        result = guard_haystack_generator(gen).run([FakeChatMessage.from_user(EMAIL_TEXT)])
        assert gen.seen[0][0].text == "Contact [EMAIL] about the invoice."
        assert "[EMAIL]" in result["replies"][0].text
        assert "bob@x.io" not in result["replies"][0].text

    async def test_run_async_redacts(self):
        gen = FakeGenerator(response=PII_RESPONSE)
        result = await guard_haystack_generator(gen).run_async(
            [FakeChatMessage.from_user(EMAIL_TEXT)]
        )
        assert gen.seen[0][0].text == "Contact [EMAIL] about the invoice."
        assert "bob@x.io" not in result["replies"][0].text

    def test_system_message_redacted(self):
        gen = FakeGenerator()
        guard_haystack_generator(gen).run([FakeChatMessage.from_system(EMAIL_TEXT)])
        assert gen.seen[0][0].text == "Contact [EMAIL] about the invoice."
        assert gen.seen[0][0].role == ChatRole.SYSTEM  # role preserved through reconstruction

    def test_non_text_content_passthrough(self):
        gen = FakeGenerator()
        tool_msg = FakeToolCallContent()
        guard_haystack_generator(gen).run([tool_msg])
        assert gen.seen[0][0] is tool_msg  # unmodified, not screened

    def test_records_usage_and_model(self):
        tracker = CostTracker()
        gen = FakeGenerator(model="fake-generator")
        guarded = guard_haystack_generator(gen, tracker=tracker, pricing={"fake": 0.00001})
        guarded.run([FakeChatMessage.from_user("hi")])
        (record,) = tracker.records
        assert record.provider == "haystack" and record.model == "fake-generator"
        assert (record.input_tokens, record.output_tokens) == (10, 5)
        assert record.estimated is False

    def test_block_on_raises(self):
        gen = FakeGenerator()
        with pytest.raises(ComplianceViolationError):
            guard_haystack_generator(gen, block_on=("email",)).run(
                [FakeChatMessage.from_user(EMAIL_TEXT)]
            )

    def test_budget_raises_before_next_call(self):
        budget = Budget(max_cost=0.0000001)
        gen = FakeGenerator()
        guarded = guard_haystack_generator(
            gen, tracker=CostTracker(), budget=budget, pricing={"fake": 0.00001}
        )
        guarded.run([FakeChatMessage.from_user("first")])
        with pytest.raises(BudgetExceededError):
            guarded.run([FakeChatMessage.from_user("second")])

    def test_no_run_async_raises_attribute_error(self):
        guarded = guard_haystack_generator(SyncOnlyGenerator())
        with pytest.raises(AttributeError):
            asyncio.run(guarded.run_async([FakeChatMessage.from_user("hi")]))

    def test_arbitrary_attrs_proxied(self):
        gen = FakeGenerator(model="custom-model")
        guarded = guard_haystack_generator(gen)
        assert guarded.model == "custom-model"  # not part of the wrapper; proxied

    def test_is_not_a_component_and_stays_a_plain_wrapper(self):
        guarded = guard_haystack_generator(FakeGenerator())
        assert isinstance(guarded, GuardedHaystackGenerator)
        assert not hasattr(guarded, "__haystack_input__")


class TestImportSafety:
    def test_module_always_importable(self):
        # haystack.py never imports haystack, so it is always safe to import,
        # even in an environment where haystack-ai was never installed.
        import multimind.integrations.frameworks.haystack as mod

        assert callable(mod.guard_haystack_generator)

    def test_lazy_export_resolves_without_haystack_installed(self):
        import multimind.integrations.frameworks as fw

        assert callable(fw.guard_haystack_generator)
        assert fw.GuardedHaystackGenerator is not None
