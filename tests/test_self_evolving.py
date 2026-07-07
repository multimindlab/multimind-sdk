"""Tests for SelfEvolvingAgent — mocked model, no live API calls."""

from __future__ import annotations

import pytest

from multimind.agents.self_evolving import SelfEvolvingAgent


class FakeModel:
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.response


def test_new_exemplar_starts_at_initial_utility():
    agent = SelfEvolvingAgent(FakeModel(), "base", initial_utility=0.5, learning_rate=0.3)

    agent.record_feedback("task A", "response A", success=True)

    exemplar = agent.top_exemplars()[0]
    assert exemplar.utility == pytest.approx(0.5 + 0.3 * 0.5)
    assert exemplar.success_count == 1


def test_repeated_success_promotes_utility_toward_one():
    agent = SelfEvolvingAgent(FakeModel(), "base", learning_rate=0.5)

    for _ in range(5):
        agent.record_feedback("task A", "good response", success=True)

    exemplar = agent.top_exemplars()[0]
    assert exemplar.utility > 0.9
    assert exemplar.success_count == 5


def test_failure_demotes_utility_toward_zero():
    agent = SelfEvolvingAgent(FakeModel(), "base", learning_rate=0.5, eviction_utility=0.0)

    agent.record_feedback("task A", "resp", success=True)
    utility_after_success = agent.top_exemplars()[0].utility

    agent.record_feedback("task A", "resp", success=False)
    utility_after_failure = agent.top_exemplars()[0].utility

    assert utility_after_failure < utility_after_success
    assert agent.top_exemplars()[0].failure_count == 1


def test_low_utility_exemplar_evicted():
    agent = SelfEvolvingAgent(
        FakeModel(), "base", learning_rate=0.9, initial_utility=0.1, eviction_utility=0.05
    )

    agent.record_feedback("task A", "resp", success=False)

    assert agent.top_exemplars() == []


def test_same_task_updates_existing_exemplar_not_duplicate():
    agent = SelfEvolvingAgent(FakeModel(), "base")

    agent.record_feedback("Task A", "first", success=True)
    agent.record_feedback("task a  ", "second", success=True)  # normalized to same key

    exemplars = agent.top_exemplars()
    assert len(exemplars) == 1
    assert exemplars[0].response == "second"
    assert exemplars[0].success_count == 2


def test_cap_enforcement_limits_top_exemplars():
    agent = SelfEvolvingAgent(FakeModel(), "base", max_examples=2)

    for i in range(5):
        agent.record_feedback(f"task {i}", f"response {i}", success=True)

    top = agent.top_exemplars()
    assert len(top) == 2


def test_top_exemplars_ranked_by_utility_descending():
    agent = SelfEvolvingAgent(FakeModel(), "base", max_examples=10, learning_rate=0.3)

    agent.record_feedback("high", "r1", success=True)
    for _ in range(3):
        agent.record_feedback("high", "r1", success=True)
    agent.record_feedback("low", "r2", success=True)  # only one promotion

    top = agent.top_exemplars()
    assert top[0].task == "high"
    assert top[0].utility > top[1].utility


def test_build_prompt_includes_base_prompt_and_exemplars():
    agent = SelfEvolvingAgent(FakeModel(), "You are a helpful assistant.", max_examples=5)
    agent.record_feedback("2+2?", "4", success=True)

    prompt = agent.build_prompt("3+3?")

    assert "You are a helpful assistant." in prompt
    assert "Task: 2+2?" in prompt
    assert "Response: 4" in prompt
    assert "Task: 3+3?" in prompt
    assert prompt.strip().endswith("Response:")


def test_build_prompt_no_exemplars_yet():
    agent = SelfEvolvingAgent(FakeModel(), "base prompt")

    prompt = agent.build_prompt("first task")

    assert "base prompt" in prompt
    assert "Examples of successful" not in prompt
    assert "Task: first task" in prompt


def test_build_prompt_only_uses_top_k_exemplars():
    agent = SelfEvolvingAgent(FakeModel(), "base", max_examples=1, learning_rate=0.5)
    agent.record_feedback("weak", "resp weak", success=True)
    for _ in range(3):
        agent.record_feedback("strong", "resp strong", success=True)

    prompt = agent.build_prompt("new task")

    assert "resp strong" in prompt
    assert "resp weak" not in prompt


@pytest.mark.asyncio
async def test_run_calls_model_with_assembled_prompt():
    model = FakeModel(response="the answer")
    agent = SelfEvolvingAgent(model, "base", max_examples=5)
    agent.record_feedback("past task", "past response", success=True)

    result = await agent.run("new task")

    assert result == "the answer"
    assert "past task" in model.prompts[0]
    assert "new task" in model.prompts[0]


def test_get_stats():
    agent = SelfEvolvingAgent(FakeModel(), "base")
    agent.record_feedback("a", "1", success=True)
    agent.record_feedback("b", "2", success=True)

    stats = agent.get_stats()

    assert stats["total_exemplars"] == 2
    assert 0.0 < stats["avg_utility"] <= 1.0
    assert stats["max_examples"] == agent.max_examples
