"""Synthetic Q/A generation for RAG fine-tuning data.

``SyntheticQAGenerator`` turns document chunks into instruction-tuning pairs:
an LLM proposes Q/A pairs per chunk (strict-parsed JSON), duplicate questions
are dropped, and each remaining pair is quality-filtered by grounding its
answer in the source chunk (reusing
:class:`multimind.evaluation.hallucination.HallucinationDetector`). No torch
is required — this is pure Python plus the caller-supplied LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..evaluation.hallucination import HallucinationDetector

_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

_DEFAULT_PROMPT_TEMPLATE = (
    "Generate {num_questions} question/answer pairs strictly grounded in the "
    "following text. Each answer must be fully supported by the text — do not "
    "add outside information.\n\n"
    "Text:\n{chunk}\n\n"
    'Respond with only a JSON array of the form: [{{"question": "...", "answer": "..."}}]'
)


@dataclass
class QAPair:
    """A generated question/answer pair grounded in a source chunk."""

    question: str
    answer: str
    source_chunk: str
    grounding_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_instruction_record(self) -> Dict[str, str]:
        """Alpaca-style instruction-tuning record."""
        return {"instruction": self.question, "input": "", "output": self.answer}


def _question_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SyntheticQAGenerator:
    """Generates grounded Q/A pairs from document chunks for instruction tuning.

    Args:
        model: LLM with an async ``generate(prompt) -> str`` method.
        num_questions_per_chunk: Q/A pairs requested per chunk.
        grounding_threshold: minimum :meth:`HallucinationDetector.check_grounding`
            score for a pair to be kept.
        prompt_template: overrides the default generation prompt; must contain
            ``{chunk}`` and ``{num_questions}`` placeholders.
    """

    def __init__(
        self,
        model: Any,
        num_questions_per_chunk: int = 3,
        grounding_threshold: float = 0.5,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
        detector: Optional[HallucinationDetector] = None,
    ):
        self.model = model
        self.num_questions_per_chunk = num_questions_per_chunk
        self.grounding_threshold = grounding_threshold
        self.prompt_template = prompt_template
        self.detector = detector or HallucinationDetector()

    @staticmethod
    def _chunk_text(chunk: Union[str, Dict[str, Any]]) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict) and "content" in chunk:
            return str(chunk["content"])
        raise ValueError(f"Each chunk must be a string or a dict with 'content', got {chunk!r}")

    def _parse_pairs(self, raw: str) -> List[Dict[str, str]]:
        match = _JSON_LIST_RE.search(str(raw))
        if not match:
            raise ValueError(f"Could not parse a Q/A list from model output: {raw!r}")
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {raw!r}") from exc
        pairs = []
        for item in items:
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                raise ValueError(f"Malformed Q/A entry in model output: {item!r}")
            pairs.append({"question": str(item["question"]), "answer": str(item["answer"])})
        return pairs

    async def generate(self, chunks: List[Union[str, Dict[str, Any]]]) -> List[QAPair]:
        """Generate, dedup and quality-filter Q/A pairs across all chunks."""
        seen_hashes = set()
        results: List[QAPair] = []
        for chunk in chunks:
            content = self._chunk_text(chunk)
            prompt = self.prompt_template.format(
                chunk=content, num_questions=self.num_questions_per_chunk
            )
            raw = await self.model.generate(prompt)
            for pair in self._parse_pairs(raw):
                question, answer = pair["question"].strip(), pair["answer"].strip()
                if not question or not answer:
                    continue
                q_hash = _question_hash(question)
                if q_hash in seen_hashes:
                    continue
                seen_hashes.add(q_hash)

                report = self.detector.check_grounding(answer, [content])
                if report.score < self.grounding_threshold:
                    continue

                results.append(
                    QAPair(
                        question=question,
                        answer=answer,
                        source_chunk=content,
                        grounding_score=report.score,
                    )
                )
        return results

    @staticmethod
    def export_jsonl(pairs: List[QAPair], path: Union[str, Path]) -> None:
        """Write pairs as Alpaca-style instruction-tuning JSONL."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_instruction_record()) + "\n")
