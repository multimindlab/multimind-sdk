"""Tests for SyntheticQAGenerator — scripted LLM output, no live API calls."""

from __future__ import annotations

import json

import pytest

from multimind.fine_tuning.synthetic_data import QAPair, SyntheticQAGenerator


class ScriptedModel:
    """Returns queued responses from generate(), in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


def qa_json(pairs):
    return json.dumps([{"question": q, "answer": a} for q, a in pairs])


CHUNK = "Python was created by Guido van Rossum and first released in 1991."


@pytest.mark.asyncio
async def test_generate_grounded_pairs_kept():
    model = ScriptedModel(
        [qa_json([("Who created Python?", "Python was created by Guido van Rossum.")])]
    )
    generator = SyntheticQAGenerator(model, grounding_threshold=0.5)

    pairs = await generator.generate([CHUNK])

    assert len(pairs) == 1
    assert pairs[0].question == "Who created Python?"
    assert pairs[0].answer == "Python was created by Guido van Rossum."
    assert pairs[0].grounding_score >= 0.5
    assert pairs[0].source_chunk == CHUNK


@pytest.mark.asyncio
async def test_ungrounded_answer_filtered_out():
    model = ScriptedModel(
        [qa_json([("Who created Python?", "It was created by aliens from Mars in 1200 BC.")])]
    )
    generator = SyntheticQAGenerator(model, grounding_threshold=0.5)

    pairs = await generator.generate([CHUNK])

    assert pairs == []


@pytest.mark.asyncio
async def test_dedup_by_question_hash_across_chunks():
    model = ScriptedModel(
        [
            qa_json([("Who created Python?", "Python was created by Guido van Rossum.")]),
            qa_json(
                [
                    ("who created python?", "Python was created by Guido van Rossum in 1991."),
                    ("When was Python released?", "Python was first released in 1991."),
                ]
            ),
        ]
    )
    generator = SyntheticQAGenerator(model, grounding_threshold=0.3)

    pairs = await generator.generate([CHUNK, CHUNK])

    questions = [p.question for p in pairs]
    assert questions.count("Who created Python?") == 1
    assert "When was Python released?" in questions
    assert len(pairs) == 2


@pytest.mark.asyncio
async def test_dict_chunks_supported():
    model = ScriptedModel(
        [qa_json([("Who created Python?", "Python was created by Guido van Rossum.")])]
    )
    generator = SyntheticQAGenerator(model, grounding_threshold=0.5)

    pairs = await generator.generate([{"content": CHUNK, "id": "c1"}])

    assert len(pairs) == 1
    assert pairs[0].source_chunk == CHUNK


@pytest.mark.asyncio
async def test_malformed_model_output_raises():
    model = ScriptedModel(["not json at all"])
    generator = SyntheticQAGenerator(model)

    with pytest.raises(ValueError, match="Could not parse"):
        await generator.generate([CHUNK])


@pytest.mark.asyncio
async def test_malformed_entry_raises():
    model = ScriptedModel([json.dumps([{"question": "Q only"}])])
    generator = SyntheticQAGenerator(model)

    with pytest.raises(ValueError, match="Malformed"):
        await generator.generate([CHUNK])


@pytest.mark.asyncio
async def test_invalid_chunk_shape_raises():
    generator = SyntheticQAGenerator(ScriptedModel([]))

    with pytest.raises(ValueError, match="content"):
        await generator.generate([{"id": "x"}])


def test_export_jsonl_alpaca_format(tmp_path):
    pairs = [
        QAPair(question="Q1", answer="A1", source_chunk="chunk1", grounding_score=0.9),
        QAPair(question="Q2", answer="A2", source_chunk="chunk2", grounding_score=0.8),
    ]
    out_path = tmp_path / "qa.jsonl"

    SyntheticQAGenerator.export_jsonl(pairs, out_path)

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record == {"instruction": "Q1", "input": "", "output": "A1"}
    record2 = json.loads(lines[1])
    assert record2 == {"instruction": "Q2", "input": "", "output": "A2"}


@pytest.mark.asyncio
async def test_multiple_chunks_isolated_grounding():
    other_chunk = "Rust is a systems programming language focused on safety."
    model = ScriptedModel(
        [
            qa_json([("Who created Python?", "Python was created by Guido van Rossum.")]),
            qa_json([("What is Rust?", "Rust is a systems programming language.")]),
        ]
    )
    generator = SyntheticQAGenerator(model, grounding_threshold=0.5)

    pairs = await generator.generate([CHUNK, other_chunk])

    assert len(pairs) == 2
    assert {p.source_chunk for p in pairs} == {CHUNK, other_chunk}
