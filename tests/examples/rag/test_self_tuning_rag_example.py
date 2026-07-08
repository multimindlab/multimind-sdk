"""
Tests for the self_tuning_rag_example module.
"""

import sys
from pathlib import Path

import pytest

# The example builds a HybridRetriever whose default sparse retriever needs sklearn
pytest.importorskip("sklearn", reason="HybridRetriever default sparse retriever needs sklearn")

from multimind.patterns.advanced_patterns import SelfImprovingRAG  # noqa: E402

# Ensure project root on sys.path for example imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from examples.rag.self_tuning_rag_example import (  # noqa: E402  pylint: disable=wrong-import-position
    MockLLM,
    MockMemory,
    MockPEFTTuner,
    MockRetriever,
    main,
)


@pytest.mark.asyncio
async def test_self_tuning_example_main_completes_without_errors():
    """Ensure the example's main coroutine executes end-to-end."""
    await main()


@pytest.mark.asyncio
async def test_process_query_returns_response_and_metadata():
    """SelfImprovingRAG should produce the model's response and metadata dict."""
    rag = SelfImprovingRAG(
        model=MockLLM(),
        retriever=MockRetriever(),
        memory=MockMemory(),
        peft_tuner=MockPEFTTuner(),
        retrain_threshold=0.8,
        retrain_window=3,
        retrain_cooldown=0,
    )

    response, metadata = await rag.process_query("Explain self tuning RAG.")

    assert response == "mocked LLM response"
    assert isinstance(metadata, dict)


@pytest.mark.asyncio
async def test_feedback_analysis_counts_positive_and_negative_feedback():
    """Feedback analytics should reflect submitted thumbs up/down counts."""
    rag = SelfImprovingRAG(
        model=MockLLM(),
        retriever=MockRetriever(),
        memory=MockMemory(),
        retrain_threshold=0.5,
        retrain_window=2,
        retrain_cooldown=0,
    )

    rag.submit_feedback("q1", "resp1", {"thumbs": "up"})
    rag.submit_feedback("q2", "resp2", {"thumbs": "down"})
    rag.submit_feedback("q3", "resp3", {})  # neutral feedback

    analytics = await rag.analyze_feedback()
    stats = analytics["stats"]

    assert stats["total_feedbacks"] == 3
    assert stats["positive"] == 1
    assert stats["negative"] == 1
    assert 0.3 < stats["average_quality"] < 0.8  # ensures neutral feedback counted


def test_negative_feedback_triggers_retraining_when_threshold_breached():
    """PEFT tuner should retrain once enough low-quality feedback accumulates."""

    class TrackingPEFTTuner(MockPEFTTuner):
        def __init__(self):
            super().__init__()
            self.trained_with = None
            self.saved = False

        def train(self, train_data):
            self.trained_with = train_data

        def save_model(self):
            self.saved = True

    tuner = TrackingPEFTTuner()
    rag = SelfImprovingRAG(
        model=MockLLM(),
        retriever=MockRetriever(),
        memory=MockMemory(),
        peft_tuner=tuner,
        retrain_threshold=0.9,
        retrain_window=3,
        retrain_cooldown=0,
    )

    for i in range(3):
        rag.submit_feedback(f"q{i}", f"resp{i}", {"thumbs": "down"})

    assert tuner.trained_with is not None
    assert len(tuner.trained_with) == 3
    assert tuner.saved is True
