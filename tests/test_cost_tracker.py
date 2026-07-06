"""Tests for cost tracking (multimind.observability.cost_tracker)."""

import json
import logging
import threading

import pytest

from multimind.observability.cost_tracker import (
    Budget,
    BudgetExceededError,
    CostTracker,
    TrackedModel,
    cost_summary,
    estimate_tokens,
    get_default_tracker,
    load_tracker,
    reset_default_tracker,
    track_costs,
)


class MockModel:
    """Minimal BaseLLM-compatible async model with blended pricing."""

    PROVIDER_NAME = "MockProvider"

    def __init__(self, response="four word mock reply", cost_per_token=0.00001, chunks=None):
        self.model_name = "mock-model"
        self.cost_per_token = cost_per_token
        self.response = response
        self.chunks = chunks or ["chunk one ", "chunk two"]
        self.calls = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        return self.response

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return self.response

    async def generate_stream(self, prompt, **kwargs):
        self.calls += 1
        for chunk in self.chunks:
            yield chunk

    async def chat_stream(self, messages, **kwargs):
        self.calls += 1
        for chunk in self.chunks:
            yield chunk

    def generate_sync(self, prompt, **kwargs):
        self.calls += 1
        return self.response

    def chat_sync(self, messages, **kwargs):
        self.calls += 1
        return self.response

    def custom_method(self):
        return "custom-result"


class UsageModel(MockModel):
    """Model exposing real usage info via last_usage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}


class UnpricedModel(MockModel):
    """Model with no pricing information at all."""

    def __init__(self, **kwargs):
        super().__init__(cost_per_token=None, **kwargs)


class PricedByTableModel(MockModel):
    """Model priced only via a MODEL_PRICING class table."""

    MODEL_PRICING = {
        "mock": 0.00002,
        "mock-model": {"input": 0.00001, "output": 0.00003},
    }

    def __init__(self, **kwargs):
        super().__init__(cost_per_token=None, **kwargs)


@pytest.fixture(autouse=True)
def _fresh_default_tracker():
    reset_default_tracker()
    yield
    reset_default_tracker()


# --- estimate_tokens ----------------------------------------------------------


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 9) == 3


# --- CostTracker accounting ---------------------------------------------------


def test_record_and_totals():
    tracker = CostTracker()
    tracker.record("openai", "gpt-4o", 100, 50, 0.001)
    tracker.record("openai", "gpt-4o-mini", 200, 100, 0.0005)
    tracker.record("claude", "claude-3", 10, 20, 0.0002, tag="batch")
    assert tracker.total_cost == pytest.approx(0.0017)
    assert tracker.total_tokens == 480

    by_model = tracker.by_model()
    assert by_model["gpt-4o"]["calls"] == 1
    assert by_model["gpt-4o"]["cost"] == pytest.approx(0.001)
    assert by_model["gpt-4o-mini"]["input_tokens"] == 200

    by_provider = tracker.by_provider()
    assert by_provider["openai"]["calls"] == 2
    assert by_provider["openai"]["cost"] == pytest.approx(0.0015)

    by_tag = tracker.by_tag()
    assert by_tag["batch"]["calls"] == 1
    assert by_tag["untagged"]["calls"] == 2


def test_summary_and_report():
    tracker = CostTracker()
    tracker.record("openai", "gpt-4o", 100, 50, 0.001, estimated=True)
    tracker.record("mock", "m1", 10, 10, 0.0, unpriced=True)
    summary = tracker.summary()
    assert summary["calls"] == 2
    assert summary["total_cost"] == pytest.approx(0.001)
    assert summary["estimated_calls"] == 1
    assert summary["unpriced_calls"] == 1
    report = tracker.report()
    assert "gpt-4o" in report
    assert "TOTAL" in report
    assert "estimated" in report
    assert "unpriced" in report


def test_report_empty_tracker():
    assert "TOTAL" in CostTracker().report()


# --- wrapper ------------------------------------------------------------------


async def test_wrapper_counts_calls_and_costs():
    tracker = CostTracker()
    model = MockModel()
    tracked = track_costs(model, tracker=tracker)
    assert isinstance(tracked, TrackedModel)

    await tracked.generate("hello world prompt")
    await tracked.chat([{"role": "user", "content": "hi there"}])
    chunks = [c async for c in tracked.generate_stream("stream prompt")]
    assert chunks == model.chunks
    async for _ in tracked.chat_stream([{"role": "user", "content": "hi"}]):
        pass

    assert model.calls == 4
    records = tracker.records
    assert len(records) == 4
    assert [r.method for r in records] == ["generate", "chat", "generate_stream", "chat_stream"]
    assert all(r.provider == "MockProvider" for r in records)
    assert all(r.model == "mock-model" for r in records)
    assert all(r.cost > 0 for r in records)
    # blended pricing: (input + output) * cost_per_token
    r0 = records[0]
    assert r0.cost == pytest.approx((r0.input_tokens + r0.output_tokens) * 0.00001)


def test_wrapper_sync_methods():
    tracker = CostTracker()
    tracked = track_costs(MockModel(), tracker=tracker, tag="sync")
    tracked.generate_sync("prompt text")
    tracked.chat_sync([{"role": "user", "content": "question"}])
    records = tracker.records
    assert [r.method for r in records] == ["generate_sync", "chat_sync"]
    assert all(r.tag == "sync" for r in records)


async def test_wrapper_delegates_other_attributes():
    tracked = track_costs(MockModel())
    assert tracked.custom_method() == "custom-result"
    assert tracked.model_name == "mock-model"


# --- estimated / unpriced flags -----------------------------------------------


async def test_estimated_flag_when_no_usage_info():
    tracker = CostTracker()
    tracked = track_costs(MockModel(response="abcdefgh"), tracker=tracker)
    await tracked.generate("x" * 40)
    record = tracker.records[0]
    assert record.estimated is True
    assert record.input_tokens == 10
    assert record.output_tokens == 2


async def test_real_usage_preferred_over_estimate():
    tracker = CostTracker()
    tracked = track_costs(UsageModel(), tracker=tracker)
    await tracked.generate("short")
    record = tracker.records[0]
    assert record.estimated is False
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.cost == pytest.approx(150 * 0.00001)


async def test_unpriced_flag_records_zero_cost():
    tracker = CostTracker()
    tracked = track_costs(UnpricedModel(), tracker=tracker)
    await tracked.generate("hello")
    record = tracker.records[0]
    assert record.unpriced is True
    assert record.cost == 0.0


async def test_model_pricing_table_prefix_match():
    tracker = CostTracker()
    tracked = track_costs(PricedByTableModel(), tracker=tracker)
    await tracked.generate("hello")
    record = tracker.records[0]
    assert record.unpriced is False
    # longest prefix "mock-model" wins and uses input/output prices
    expected = record.input_tokens * 0.00001 + record.output_tokens * 0.00003
    assert record.cost == pytest.approx(expected)


# --- budget -------------------------------------------------------------------


async def test_budget_raises_before_next_call():
    tracker = CostTracker()
    model = MockModel(cost_per_token=1.0)  # each call costs several dollars
    budget = Budget(max_cost=1.0)
    tracked = track_costs(model, tracker=tracker, budget=budget)

    await tracked.generate("this call goes through")
    assert model.calls == 1
    assert budget.spent > budget.max_cost

    with pytest.raises(BudgetExceededError) as excinfo:
        await tracked.generate("this one is blocked")
    assert model.calls == 1  # blocked before dispatch
    assert excinfo.value.max_cost == 1.0


async def test_budget_soft_warning_at_threshold(caplog):
    model = MockModel(cost_per_token=0.1)
    budget = Budget(max_cost=1000.0, warn_ratio=0.0001)
    tracked = track_costs(model, tracker=CostTracker(), budget=budget)
    with caplog.at_level(logging.WARNING, logger="multimind.observability.cost_tracker"):
        await tracked.generate("prompt")
        await tracked.generate("prompt")
    warnings = [r for r in caplog.records if "Budget at" in r.message]
    assert len(warnings) == 1  # warned once, not per call


def test_budget_validation():
    with pytest.raises(ValueError):
        Budget(max_cost=0)
    with pytest.raises(ValueError):
        Budget(max_cost=1.0, period="monthly")
    with pytest.raises(ValueError):
        Budget(max_cost=1.0, warn_ratio=2.0)


def test_budget_reset():
    budget = Budget(max_cost=1.0)
    budget.add(2.0)
    with pytest.raises(BudgetExceededError):
        budget.check()
    budget.reset()
    budget.check()
    assert budget.remaining == 1.0


# --- persistence --------------------------------------------------------------


def test_jsonl_persistence_no_content(tmp_path):
    path = tmp_path / "costs.jsonl"
    tracker = CostTracker(jsonl_path=path)
    tracker.record("openai", "gpt-4o", 100, 50, 0.001, tag="job-1")
    tracker.record("mock", "m1", 1, 2, 0.0, unpriced=True)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["model"] == "gpt-4o"
    assert first["input_tokens"] == 100
    assert first["tag"] == "job-1"
    assert "timestamp" in first
    # never persists prompt/response content
    for line in lines:
        entry = json.loads(line)
        assert "prompt" not in entry
        assert "content" not in entry
        assert "response" not in entry


async def test_wrapper_persists_records(tmp_path):
    path = tmp_path / "wrapped.jsonl"
    tracker = CostTracker(jsonl_path=path)
    tracked = track_costs(MockModel(), tracker=tracker)
    await tracked.generate("hello")
    entry = json.loads(path.read_text().strip())
    assert entry["method"] == "generate"
    assert entry["estimated"] is True


# --- chargeback ---------------------------------------------------------------


def test_chargeback_math_and_untagged():
    tracker = CostTracker()
    tracker.record("openai", "gpt-4o", 100, 50, 0.003, tag="team-a")
    tracker.record("openai", "gpt-4o", 10, 5, 0.001, tag="team-a", estimated=True)
    tracker.record("claude", "claude-3", 20, 10, 0.006, tag="team-b")
    tracker.record("mock", "m1", 1, 1, 0.0, unpriced=True)
    data = tracker.chargeback()
    assert data["calls"] == 4
    assert data["total_cost"] == pytest.approx(0.01)
    by_tag = data["by_tag"]
    assert set(by_tag) == {"team-a", "team-b", "(untagged)"}
    a = by_tag["team-a"]
    assert a["calls"] == 2
    assert a["cost"] == pytest.approx(0.004)
    assert a["total_tokens"] == 165
    assert a["estimated_calls"] == 1
    assert a["share_pct"] == pytest.approx(40.0)
    assert by_tag["team-b"]["share_pct"] == pytest.approx(60.0)
    assert by_tag["(untagged)"]["unpriced_calls"] == 1
    assert sum(g["share_pct"] for g in by_tag.values()) == pytest.approx(100.0)


def test_chargeback_zero_cost_shares():
    tracker = CostTracker()
    tracker.record("mock", "m1", 1, 1, 0.0, unpriced=True)
    data = tracker.chargeback()
    assert data["total_cost"] == 0.0
    assert data["by_tag"]["(untagged)"]["share_pct"] == 0.0


def test_chargeback_period_filter():
    tracker = CostTracker()
    tracker.record("p", "m", 1, 1, 0.001, tag="june")
    tracker.record("p", "m", 1, 1, 0.002, tag="july")
    records = tracker.records
    records[0].timestamp = "2026-06-15T00:00:00+00:00"
    records[1].timestamp = "2026-07-01T00:00:00+00:00"
    data = tracker.chargeback(period="2026-07")
    assert data["calls"] == 1
    assert set(data["by_tag"]) == {"july"}
    assert data["by_tag"]["july"]["share_pct"] == pytest.approx(100.0)


def test_report_chargeback_table():
    tracker = CostTracker()
    tracker.record("openai", "gpt-4o", 100, 50, 0.003, tag="team-a")
    tracker.record("mock", "m1", 1, 1, 0.0, unpriced=True)
    report = tracker.report_chargeback()
    assert "Chargeback report" in report
    assert "team-a" in report
    assert "(untagged)" in report
    assert "TOTAL" in report
    assert "unpriced" in report


def test_load_tracker_round_trip(tmp_path):
    path = tmp_path / "costs.jsonl"
    tracker = CostTracker(jsonl_path=path)
    tracker.record(
        "openai", "gpt-4o", 100, 50, 0.001, tag="job-1", method="generate", estimated=True
    )
    tracker.record("mock", "m1", 1, 2, 0.0, unpriced=True)
    loaded = load_tracker(path)
    assert [r.to_dict() for r in loaded.records] == [r.to_dict() for r in tracker.records]
    assert loaded.chargeback() == tracker.chargeback()
    # loaded tracker is detached from the file
    loaded.record("p", "m", 1, 1, 0.001)
    assert len(path.read_text().strip().splitlines()) == 2


# --- thread safety ------------------------------------------------------------


def test_concurrent_records_thread_safety():
    tracker = CostTracker()

    def worker():
        for _ in range(200):
            tracker.record("p", "m", 1, 1, 0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tracker.records) == 1600
    assert tracker.total_tokens == 3200
    assert tracker.total_cost == pytest.approx(1.6)


# --- default tracker ----------------------------------------------------------


async def test_default_tracker_and_cost_summary(capsys):
    tracked = track_costs(MockModel())
    await tracked.generate("hello")
    default = get_default_tracker()
    assert len(default.records) == 1
    summary = cost_summary()
    captured = capsys.readouterr()
    assert "Cost report" in captured.out
    assert summary["calls"] == 1
