"""Hallucination detection: grounding checks and a drop-in model guard.

Two complementary modes:

* Grounding check (default, no LLM needed): the response is split into
  sentences and each sentence is scored against source snippets by n-gram
  containment (ROUGE-style precision from :mod:`.metrics`). Deterministic,
  fast, works offline. Being lexical, it cannot catch paraphrased
  contradictions — use the judge mode for that.
* LLM-judge check: a judge model extracts factual claims from the response
  and rates each one supported/contradicted/unverifiable against the sources.

:func:`detect_hallucinations` wraps any BaseLLM-compatible model (duck-typed
async ``generate`` / ``chat`` / ``generate_stream``) so every generation is
grounding-checked against static or per-prompt (RAG) sources.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .metrics import rouge_n, tokenize

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

_VERDICT_WEIGHTS = {"supported": 1.0, "unverifiable": 0.5, "contradicted": 0.0}
_ON_FLAG_MODES = ("annotate", "raise", "retry")

_GROUNDING_INSTRUCTION = (
    "Answer strictly using the provided sources. Do not state any fact that "
    "is not supported by them; if the sources do not cover something, say so."
)

SourcesArg = Optional[Union[List[str], Callable[[str], List[str]]]]


class HallucinationError(Exception):
    """Raised by a ``on_flag="raise"`` guard when grounding falls below threshold."""

    def __init__(self, message: str, report: Optional["GroundingReport"] = None):
        super().__init__(message)
        self.report = report


@dataclass
class SentenceGrounding:
    """Grounding verdict for a single response sentence."""

    sentence: str
    verdict: str  # "supported" | "partial" | "unsupported"
    score: float
    best_source_snippet: Optional[str]


@dataclass
class GroundingReport:
    """Sentence-level grounding of a response against its sources."""

    score: float
    sentences: List[SentenceGrounding]
    unsupported_count: int

    def summary(self) -> str:
        counts = {"supported": 0, "partial": 0, "unsupported": 0}
        for s in self.sentences:
            counts[s.verdict] += 1
        return (
            f"grounding score {self.score:.2f} over {len(self.sentences)} sentence(s): "
            f"{counts['supported']} supported, {counts['partial']} partial, "
            f"{counts['unsupported']} unsupported"
        )


@dataclass
class Claim:
    """A single factual claim extracted by the judge, with its verdict."""

    claim: str
    verdict: str  # "supported" | "contradicted" | "unverifiable"


@dataclass
class ClaimReport:
    """Claim-level verdicts from an LLM judge."""

    score: float
    claims: List[Claim] = field(default_factory=list)

    def summary(self) -> str:
        counts = {v: 0 for v in _VERDICT_WEIGHTS}
        for c in self.claims:
            counts[c.verdict] += 1
        return (
            f"claim score {self.score:.2f} over {len(self.claims)} claim(s): "
            f"{counts['supported']} supported, {counts['contradicted']} contradicted, "
            f"{counts['unverifiable']} unverifiable"
        )


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation and newlines."""
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _snippets(source: str) -> List[str]:
    # Single sentences plus adjacent pairs, so facts spanning two source
    # sentences still find a containing snippet.
    sentences = split_sentences(source)
    return sentences + [f"{sentences[i]} {sentences[i + 1]}" for i in range(len(sentences) - 1)]


def _containment(sentence: str, snippet: str) -> float:
    """Fraction of the sentence's n-grams found in the snippet (unigram + bigram)."""
    p1 = rouge_n(sentence, snippet, 1)["precision"]
    if len(tokenize(sentence)) < 2:
        return p1
    p2 = rouge_n(sentence, snippet, 2)["precision"]
    return 0.6 * p1 + 0.4 * p2


class HallucinationDetector:
    """Scores how well a response is grounded in a set of source texts."""

    def __init__(
        self,
        judge_model: Optional[Any] = None,
        supported_threshold: float = 0.7,
        partial_threshold: float = 0.4,
    ):
        self.judge_model = judge_model
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold

    def check_grounding(self, response: str, sources: List[str]) -> GroundingReport:
        """Deterministic lexical grounding check; no model call.

        An empty response yields a perfect score (nothing to hallucinate);
        empty sources leave every sentence unsupported with score 0.
        """
        snippets = [snippet for source in sources for snippet in _snippets(source)]
        results: List[SentenceGrounding] = []
        for sentence in split_sentences(response):
            best_score, best_snippet = 0.0, None
            for snippet in snippets:
                score = _containment(sentence, snippet)
                if score > best_score:
                    best_score, best_snippet = score, snippet
            if best_score >= self.supported_threshold:
                verdict = "supported"
            elif best_score >= self.partial_threshold:
                verdict = "partial"
            else:
                verdict = "unsupported"
            results.append(SentenceGrounding(sentence, verdict, best_score, best_snippet))
        overall = sum(r.score for r in results) / len(results) if results else 1.0
        unsupported = sum(1 for r in results if r.verdict == "unsupported")
        return GroundingReport(score=overall, sentences=results, unsupported_count=unsupported)

    async def check_claims(
        self, response: str, sources: List[str], judge_model: Optional[Any] = None
    ) -> ClaimReport:
        """LLM-judge check: extract factual claims and rate each against the sources."""
        judge = judge_model or self.judge_model
        if judge is None:
            raise ValueError("check_claims requires a judge_model")

        sources_block = "\n\n".join(f"[{i}] {source}" for i, source in enumerate(sources))
        prompt = f"""
        Extract the factual claims made in the response and rate each one
        against the sources.
        Consider:
        1. Information not present in the sources
        2. Contradictions with the sources
        3. Fabricated details

        Sources:
        {sources_block}

        Response:
        {response}

        Respond with only a JSON list of objects with keys "claim" and "verdict",
        where "verdict" is one of "supported", "contradicted" or "unverifiable",
        e.g. [{{"claim": "...", "verdict": "supported"}}].
        """

        raw = await judge.generate(prompt=prompt)
        claims = self._parse_claims(raw)
        score = sum(_VERDICT_WEIGHTS[c.verdict] for c in claims) / len(claims) if claims else 1.0
        return ClaimReport(score=score, claims=claims)

    @staticmethod
    def _parse_claims(text: str) -> List[Claim]:
        match = _JSON_LIST_RE.search(str(text))
        if not match:
            raise ValueError(f"Could not parse a claim list from judge output: {text!r}")
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Judge output is not valid JSON: {text!r}") from e
        claims: List[Claim] = []
        for item in items:
            if not isinstance(item, dict) or "claim" not in item or "verdict" not in item:
                raise ValueError(f"Malformed claim entry in judge output: {item!r}")
            verdict = str(item["verdict"]).strip().lower()
            if verdict not in _VERDICT_WEIGHTS:
                raise ValueError(f"Unknown verdict in judge output: {item['verdict']!r}")
            claims.append(Claim(claim=str(item["claim"]), verdict=verdict))
        return claims


class GuardedModel:
    """Drop-in hallucination guard around any BaseLLM-compatible model.

    After every generation the output is grounding-checked against ``sources``
    (a static list, or a callable ``(prompt) -> list[str]`` for RAG). The last
    :class:`GroundingReport` is always available on ``last_report``. When the
    score falls below ``threshold``, ``on_flag`` decides what happens:
    "annotate" logs a warning, "raise" raises :class:`HallucinationError`,
    "retry" re-generates once with a grounding instruction appended and then
    annotates. All other attribute access is proxied to the wrapped model.

    Streaming: chunks are passed through untouched; the check runs on the full
    accumulated text after the stream ends ("retry" degrades to "annotate"
    since streamed text cannot be retracted).
    """

    def __init__(
        self,
        model: Any,
        sources: SourcesArg = None,
        judge_model: Optional[Any] = None,
        threshold: float = 0.5,
        on_flag: str = "annotate",
        detector: Optional[HallucinationDetector] = None,
    ):
        if on_flag not in _ON_FLAG_MODES:
            raise ValueError(f"Unknown on_flag mode: {on_flag!r} (use one of {_ON_FLAG_MODES})")
        self._model = model
        self.sources = sources
        self.threshold = threshold
        self.on_flag = on_flag
        self.detector = detector or HallucinationDetector(judge_model=judge_model)
        self.last_report: Optional[GroundingReport] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _resolve_sources(self, prompt: str) -> List[str]:
        if callable(self.sources):
            return list(self.sources(prompt))
        return list(self.sources or [])

    def _flag(self, report: GroundingReport, method: str) -> None:
        if report.score >= self.threshold:
            return
        if self.on_flag == "raise":
            raise HallucinationError(
                f"Response failed grounding check in {method}: {report.summary()} "
                f"(threshold {self.threshold:.2f})",
                report,
            )
        logger.warning(
            "Possible hallucination in %s (threshold %.2f): %s",
            method,
            self.threshold,
            report.summary(),
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        output = await self._model.generate(prompt, **kwargs)
        sources = self._resolve_sources(prompt)
        report = self.detector.check_grounding(output, sources)
        if report.score < self.threshold and self.on_flag == "retry":
            output = await self._model.generate(f"{prompt}\n\n{_GROUNDING_INSTRUCTION}", **kwargs)
            report = self.detector.check_grounding(output, sources)
        self.last_report = report
        self._flag(report, "generate")
        return output

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        output = await self._model.chat(messages, **kwargs)
        sources = self._resolve_sources(self._last_user_content(messages))
        report = self.detector.check_grounding(output, sources)
        if report.score < self.threshold and self.on_flag == "retry":
            retry_messages = list(messages) + [{"role": "user", "content": _GROUNDING_INSTRUCTION}]
            output = await self._model.chat(retry_messages, **kwargs)
            report = self.detector.check_grounding(output, sources)
        self.last_report = report
        self._flag(report, "chat")
        return output

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        collected: List[str] = []
        async for chunk in self._model.generate_stream(prompt, **kwargs):
            collected.append(chunk)
            yield chunk
        report = self.detector.check_grounding("".join(collected), self._resolve_sources(prompt))
        self.last_report = report
        self._flag(report, "generate_stream")

    @staticmethod
    def _last_user_content(messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""


def detect_hallucinations(
    model: Any,
    sources: SourcesArg = None,
    judge_model: Optional[Any] = None,
    threshold: float = 0.5,
    on_flag: str = "annotate",
    **kwargs,
) -> GuardedModel:
    """Convenience wrapper: ``detect_hallucinations(model, sources=[...]) -> GuardedModel``."""
    return GuardedModel(
        model,
        sources=sources,
        judge_model=judge_model,
        threshold=threshold,
        on_flag=on_flag,
        **kwargs,
    )
