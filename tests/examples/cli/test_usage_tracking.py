"""
Tests for usage_tracking.py CLI example.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add repo root to path so we can import examples
sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.cli.usage_tracking import main


class MockUsageTracker:
    """Mock usage tracker that records usage in memory."""

    instances = []

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_costs = {}
        self.records = []
        MockUsageTracker.instances.append(self)

    def set_model_costs(self, model: str, input_cost_per_token: float, output_cost_per_token: float):
        self.model_costs[model] = {
            "input": input_cost_per_token,
            "output": output_cost_per_token,
        }

    def track_usage(self, model: str, operation: str, input_tokens: int, output_tokens: int, metadata=None):
        self.records.append(
            {
                "model": model,
                "operation": operation,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "metadata": metadata or {},
            }
        )

    def get_usage_summary(self, start_date=None, end_date=None):
        summary = {"total_cost": 0.0, "models": {}}
        for record in self.records:
            model_entry = summary["models"].setdefault(
                record["model"],
                {
                    "total_cost": 0.0,
                    "operations": {},
                },
            )
            ops = model_entry["operations"].setdefault(
                record["operation"],
                {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                },
            )
            ops["count"] += 1
            ops["input_tokens"] += record["input_tokens"]
            ops["output_tokens"] += record["output_tokens"]
        return summary

    def export_usage(self, *args, **kwargs):
        return None


class MockTraceLogger:
    """Mock trace logger that stores events in memory."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.events = []
        self.traces = []

    def start_trace(self, trace_id: str, operation: str, metadata=None):
        self.traces.append({"trace_id": trace_id, "operation": operation, "metadata": metadata or {}})

    def add_event(self, trace_id: str, event_type: str, data=None):
        self.events.append({"trace_id": trace_id, "event_type": event_type, "data": data or {}})

    def end_trace(self, trace_id: str, status: str, result=None):
        self.events.append({"trace_id": trace_id, "event_type": "end", "status": status, "result": result or {}})


class MockOpenAIModel:
    """Mock OpenAI model."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def generate(self, prompt: str):
        return f"OpenAI response to: {prompt}"


class MockClaudeModel:
    """Mock Claude model."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def generate(self, prompt: str):
        return f"Claude response to: {prompt}"


@pytest.fixture(autouse=True)
def clear_instances():
    """Reset shared mock state before each test."""
    MockUsageTracker.instances.clear()
    yield
    MockUsageTracker.instances.clear()


@pytest.mark.asyncio
async def test_usage_tracking_imports():
    """Ensure the module imports cleanly."""
    try:
        from examples.cli import usage_tracking  # noqa: F401
    except Exception as exc:
        pytest.fail(f"Import failed: {exc}")


@pytest.mark.asyncio
async def test_main_with_openai_only(monkeypatch):
    """Run main with only OpenAI API key present."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)

    with patch("examples.cli.usage_tracking.UsageTracker", MockUsageTracker), \
         patch("examples.cli.usage_tracking.TraceLogger", MockTraceLogger), \
         patch("examples.cli.usage_tracking.OpenAIModel", MockOpenAIModel), \
         patch("examples.cli.usage_tracking.ClaudeModel", MockClaudeModel), \
         patch("examples.cli.usage_tracking.load_dotenv"):
        await main()

    tracker = MockUsageTracker.instances[0]
    assert tracker.records, "Usage should be tracked for OpenAI"
    assert all(record["model"] == "gpt-3.5-turbo" for record in tracker.records)


@pytest.mark.asyncio
async def test_main_with_claude_only(monkeypatch):
    """Run main with only Claude API key present."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("examples.cli.usage_tracking.UsageTracker", MockUsageTracker), \
         patch("examples.cli.usage_tracking.TraceLogger", MockTraceLogger), \
         patch("examples.cli.usage_tracking.OpenAIModel", MockOpenAIModel), \
         patch("examples.cli.usage_tracking.ClaudeModel", MockClaudeModel), \
         patch("examples.cli.usage_tracking.load_dotenv"):
        await main()

    tracker = MockUsageTracker.instances[0]
    assert tracker.records, "Usage should be tracked for Claude"
    assert all(record["model"] == "claude-3-sonnet" for record in tracker.records)


@pytest.mark.asyncio
async def test_main_with_both_models(monkeypatch):
    """Run main when both API keys are present."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude")

    with patch("examples.cli.usage_tracking.UsageTracker", MockUsageTracker), \
         patch("examples.cli.usage_tracking.TraceLogger", MockTraceLogger), \
         patch("examples.cli.usage_tracking.OpenAIModel", MockOpenAIModel), \
         patch("examples.cli.usage_tracking.ClaudeModel", MockClaudeModel), \
         patch("examples.cli.usage_tracking.load_dotenv"):
        await main()

    tracker = MockUsageTracker.instances[0]
    models_seen = {record["model"] for record in tracker.records}
    assert models_seen == {"gpt-3.5-turbo", "claude-3-sonnet"}

