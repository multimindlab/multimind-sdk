"""
Hallucination detection demo: offline grounding checks and the model guard.

Uses a tiny in-file fake model, so it runs end-to-end with no API keys:
    python examples/evaluation/hallucination_detection.py
"""

import asyncio

from multimind.evaluation.hallucination import (
    HallucinationDetector,
    HallucinationError,
    detect_hallucinations,
)

SOURCES = [
    "The Eiffel Tower is located in Paris. It was completed in 1889 for the "
    "World's Fair. The tower is 330 metres tall.",
    "Gustave Eiffel's company designed and built the tower.",
]

HALLUCINATED = (
    "The Eiffel Tower is located in Paris. The tower is 330 metres tall. "
    "It was designed by Leonardo da Vinci as a giant sundial."
)

GROUNDED = (
    "The Eiffel Tower is located in Paris. It was completed in 1889. The tower is 330 metres tall."
)


class FakeRAGModel:
    """Fake model that hallucinates first, then answers from the sources when
    the retry prompt carries the grounding instruction."""

    model_name = "fake-rag-model"

    async def generate(self, prompt, **kwargs):
        if "provided sources" in prompt:
            return GROUNDED
        return HALLUCINATED


async def main():
    detector = HallucinationDetector()

    print("=== 1. Grounding check on a hallucinated answer ===")
    report = detector.check_grounding(HALLUCINATED, SOURCES)
    print(f"  {report.summary()}")
    for s in report.sentences:
        print(f"  [{s.verdict:<11}] score={s.score:.2f}  {s.sentence}")
        if s.best_source_snippet:
            print(f"                best match: {s.best_source_snippet!r}")

    print("\n=== 2. Guarded model, on_flag='annotate' (default) ===")
    guarded = detect_hallucinations(FakeRAGModel(), sources=SOURCES, threshold=0.9)
    answer = await guarded.generate("Tell me about the Eiffel Tower")
    print(f"  answer: {answer}")
    print(f"  report: {guarded.last_report.summary()}")

    print("\n=== 3. on_flag='raise' ===")
    strict = detect_hallucinations(FakeRAGModel(), sources=SOURCES, threshold=0.9, on_flag="raise")
    try:
        await strict.generate("Tell me about the Eiffel Tower")
    except HallucinationError as exc:
        print(f"  Raised: {exc}")

    print("\n=== 4. on_flag='retry' (re-generate once with grounding instruction) ===")
    retrying = detect_hallucinations(
        FakeRAGModel(), sources=SOURCES, threshold=0.9, on_flag="retry"
    )
    answer = await retrying.generate("Tell me about the Eiffel Tower")
    print(f"  answer: {answer}")
    print(f"  report: {retrying.last_report.summary()}")

    print("\n=== 5. Callable sources (per-prompt retrieval, RAG-style) ===")
    guarded = detect_hallucinations(
        FakeRAGModel(),
        sources=lambda prompt: SOURCES if "Eiffel" in prompt else [],
        threshold=0.9,
    )
    await guarded.generate("Tell me about the Eiffel Tower")
    print(f"  report: {guarded.last_report.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
