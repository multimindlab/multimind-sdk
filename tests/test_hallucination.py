"""Tests for hallucination detection (multimind.evaluation.hallucination)."""

import logging

import pytest

# multimind.evaluation.__init__ pulls in advanced_evaluation/evaluation, which
# need these heavy optional deps at import time.
pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")
pytest.importorskip("sklearn")

from multimind.evaluation.hallucination import (  # noqa: E402
    Claim,
    ClaimReport,
    GroundingReport,
    GuardedModel,
    HallucinationDetector,
    HallucinationError,
    SentenceGrounding,
    detect_hallucinations,
    split_sentences,
)

SOURCES = [
    "The Eiffel Tower is located in Paris. It was completed in 1889 for the "
    "World's Fair. The tower is 330 metres tall.",
    "Gustave Eiffel's company designed and built the tower.",
]

GROUNDED = "The Eiffel Tower is located in Paris. The tower is 330 metres tall."
FABRICATED = (
    "The Eiffel Tower is located in Paris. It was designed by Leonardo da Vinci as a giant sundial."
)


class MockModel:
    """Minimal BaseLLM-compatible async model for guard tests."""

    def __init__(self, responses=None, chunks=None):
        self.model_name = "mock-model"
        self.responses = list(responses or ["ok"])
        self.chunks = chunks or []
        self.prompts = []
        self.last_messages = None

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    async def generate_stream(self, prompt, **kwargs):
        self.prompts.append(prompt)
        for chunk in self.chunks:
            yield chunk

    def custom_method(self):
        return "custom-result"


class MockJudge:
    """Judge model returning a canned response."""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None

    async def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self.response


# --- sentence splitting ---------------------------------------------------


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_newlines_and_whitespace():
    assert split_sentences("  First line\nSecond line.  ") == ["First line", "Second line."]


def test_split_sentences_empty():
    assert split_sentences("") == []


# --- grounding check: known answers ----------------------------------------


def test_fully_grounded_response_scores_high():
    report = HallucinationDetector().check_grounding(GROUNDED, SOURCES)
    assert report.score > 0.9
    assert report.unsupported_count == 0
    assert [s.verdict for s in report.sentences] == ["supported", "supported"]
    assert all(s.best_source_snippet for s in report.sentences)


def test_fabricated_sentence_flagged_unsupported():
    report = HallucinationDetector().check_grounding(FABRICATED, SOURCES)
    assert report.unsupported_count == 1
    verdicts = {s.sentence: s.verdict for s in report.sentences}
    assert verdicts["The Eiffel Tower is located in Paris."] == "supported"
    assert verdicts["It was designed by Leonardo da Vinci as a giant sundial."] == "unsupported"
    assert report.score < 0.7


def test_empty_sources_all_unsupported_score_zero():
    report = HallucinationDetector().check_grounding(GROUNDED, [])
    assert report.score == 0.0
    assert report.unsupported_count == len(report.sentences) == 2
    assert all(s.verdict == "unsupported" for s in report.sentences)
    assert all(s.best_source_snippet is None for s in report.sentences)


def test_empty_response_scores_perfect():
    report = HallucinationDetector().check_grounding("", SOURCES)
    assert report.score == 1.0
    assert report.sentences == []
    assert report.unsupported_count == 0


def test_fact_spanning_adjacent_source_sentences():
    # Merges tokens from two adjacent source sentences; pair snippets cover it
    response = "The Eiffel Tower in Paris was completed in 1889."
    report = HallucinationDetector().check_grounding(response, SOURCES)
    assert report.sentences[0].verdict == "supported"


def test_summary_counts():
    report = HallucinationDetector().check_grounding(FABRICATED, SOURCES)
    summary = report.summary()
    assert "1 supported" in summary
    assert "1 unsupported" in summary
    assert "2 sentence(s)" in summary


def test_detector_thresholds_configurable():
    strict = HallucinationDetector(supported_threshold=1.01)
    report = strict.check_grounding(GROUNDED, SOURCES)
    assert all(s.verdict != "supported" for s in report.sentences)


# --- guarded model: annotate / raise / retry --------------------------------


async def test_annotate_returns_output_and_sets_report(caplog):
    model = MockModel(responses=[FABRICATED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9)
    with caplog.at_level(logging.WARNING, logger="multimind.evaluation.hallucination"):
        result = await guarded.generate("Tell me about the Eiffel Tower")
    assert result == FABRICATED
    assert isinstance(guarded.last_report, GroundingReport)
    assert guarded.last_report.unsupported_count == 1
    assert any("Possible hallucination" in r.message for r in caplog.records)


async def test_annotate_no_warning_when_grounded(caplog):
    model = MockModel(responses=[GROUNDED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9)
    with caplog.at_level(logging.WARNING, logger="multimind.evaluation.hallucination"):
        result = await guarded.generate("Tell me about the Eiffel Tower")
    assert result == GROUNDED
    assert guarded.last_report.score > 0.9
    assert not caplog.records


async def test_raise_mode_raises_with_report():
    model = MockModel(responses=[FABRICATED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="raise")
    with pytest.raises(HallucinationError) as excinfo:
        await guarded.generate("Tell me about the Eiffel Tower")
    assert isinstance(excinfo.value.report, GroundingReport)
    assert excinfo.value.report.unsupported_count == 1
    assert guarded.last_report is excinfo.value.report


async def test_retry_mode_regenerates_with_grounding_instruction():
    model = MockModel(responses=[FABRICATED, GROUNDED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="retry")
    result = await guarded.generate("Tell me about the Eiffel Tower")
    assert result == GROUNDED
    assert len(model.prompts) == 2
    assert model.prompts[0] == "Tell me about the Eiffel Tower"
    assert "strictly using the provided sources" in model.prompts[1]
    assert guarded.last_report.score > 0.9


async def test_retry_mode_annotates_when_still_flagged(caplog):
    model = MockModel(responses=[FABRICATED, FABRICATED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="retry")
    with caplog.at_level(logging.WARNING, logger="multimind.evaluation.hallucination"):
        result = await guarded.generate("Tell me about the Eiffel Tower")
    assert result == FABRICATED
    assert len(model.prompts) == 2
    assert any("Possible hallucination" in r.message for r in caplog.records)


async def test_retry_not_triggered_when_grounded():
    model = MockModel(responses=[GROUNDED, "should not be requested"])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="retry")
    result = await guarded.generate("Tell me about the Eiffel Tower")
    assert result == GROUNDED
    assert len(model.prompts) == 1


def test_invalid_on_flag_raises():
    with pytest.raises(ValueError):
        detect_hallucinations(MockModel(), sources=SOURCES, on_flag="ignore")


async def test_callable_sources_receive_prompt():
    seen = []

    def retrieve(prompt):
        seen.append(prompt)
        return SOURCES

    model = MockModel(responses=[GROUNDED])
    guarded = detect_hallucinations(model, sources=retrieve, threshold=0.9)
    await guarded.generate("Tell me about the Eiffel Tower")
    assert seen == ["Tell me about the Eiffel Tower"]
    assert guarded.last_report.score > 0.9


async def test_chat_checks_output_and_retries_with_instruction():
    model = MockModel(responses=[FABRICATED, GROUNDED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="retry")
    messages = [{"role": "user", "content": "Tell me about the Eiffel Tower"}]
    result = await guarded.chat(messages)
    assert result == GROUNDED
    assert len(model.last_messages) == 2
    assert "strictly using the provided sources" in model.last_messages[-1]["content"]
    assert messages == [{"role": "user", "content": "Tell me about the Eiffel Tower"}]


async def test_stream_passthrough_then_checked():
    chunks = [FABRICATED[i : i + 20] for i in range(0, len(FABRICATED), 20)]
    model = MockModel(chunks=chunks)
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9)
    collected = [chunk async for chunk in guarded.generate_stream("hi")]
    assert collected == chunks  # untouched passthrough
    assert guarded.last_report.unsupported_count == 1


async def test_stream_raise_mode_raises_after_stream():
    model = MockModel(chunks=[FABRICATED])
    guarded = detect_hallucinations(model, sources=SOURCES, threshold=0.9, on_flag="raise")
    with pytest.raises(HallucinationError):
        async for _ in guarded.generate_stream("hi"):
            pass


def test_getattr_delegation():
    model = MockModel()
    guarded = detect_hallucinations(model, sources=SOURCES)
    assert guarded.model_name == "mock-model"
    assert guarded.custom_method() == "custom-result"
    assert isinstance(guarded, GuardedModel)


# --- judge mode --------------------------------------------------------------


async def test_check_claims_parses_verdicts():
    judge = MockJudge(
        'Here are the claims:\n[{"claim": "The tower is in Paris", "verdict": "Supported"},'
        ' {"claim": "It is 500m tall", "verdict": "contradicted"},'
        ' {"claim": "It is painted yearly", "verdict": "unverifiable"}]'
    )
    detector = HallucinationDetector(judge_model=judge)
    report = await detector.check_claims(FABRICATED, SOURCES)
    assert isinstance(report, ClaimReport)
    assert [c.verdict for c in report.claims] == ["supported", "contradicted", "unverifiable"]
    assert report.score == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert FABRICATED in judge.last_prompt
    assert SOURCES[0] in judge.last_prompt
    assert "3 claim(s)" in report.summary()


async def test_check_claims_empty_list_scores_perfect():
    detector = HallucinationDetector(judge_model=MockJudge("[]"))
    report = await detector.check_claims("Hello!", SOURCES)
    assert report.score == 1.0
    assert report.claims == []


async def test_check_claims_unparseable_raises():
    detector = HallucinationDetector(judge_model=MockJudge("I cannot evaluate this."))
    with pytest.raises(ValueError):
        await detector.check_claims(FABRICATED, SOURCES)


async def test_check_claims_invalid_json_raises():
    detector = HallucinationDetector(judge_model=MockJudge('[{"claim": broken]'))
    with pytest.raises(ValueError):
        await detector.check_claims(FABRICATED, SOURCES)


async def test_check_claims_unknown_verdict_raises():
    detector = HallucinationDetector(judge_model=MockJudge('[{"claim": "x", "verdict": "maybe"}]'))
    with pytest.raises(ValueError):
        await detector.check_claims(FABRICATED, SOURCES)


async def test_check_claims_without_judge_raises():
    with pytest.raises(ValueError):
        await HallucinationDetector().check_claims(FABRICATED, SOURCES)


async def test_check_claims_explicit_judge_overrides_default():
    default = MockJudge('[{"claim": "a", "verdict": "contradicted"}]')
    override = MockJudge('[{"claim": "a", "verdict": "supported"}]')
    detector = HallucinationDetector(judge_model=default)
    report = await detector.check_claims(GROUNDED, SOURCES, judge_model=override)
    assert report.score == 1.0
    assert default.last_prompt is None


# --- exports ------------------------------------------------------------------


def test_package_exports():
    from multimind import evaluation

    assert evaluation.HallucinationDetector is HallucinationDetector
    assert evaluation.HallucinationError is HallucinationError
    assert evaluation.GuardedModel is GuardedModel
    assert evaluation.GroundingReport is GroundingReport
    assert evaluation.SentenceGrounding is SentenceGrounding
    assert evaluation.Claim is Claim
    assert evaluation.ClaimReport is ClaimReport
    assert evaluation.detect_hallucinations is detect_hallucinations
