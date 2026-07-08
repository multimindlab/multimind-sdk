"""Offline tracing demo: trace a two-span chain to SQLite and print the run tree.

Runs entirely offline with a fake model — no API keys, no network. Shows the
privacy-first default: the SQLite rows hold char counts and SHA-256 hashes,
never the prompt or response text.

Usage: python examples/observability/tracing_demo.py
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path

from multimind.observability import RunTracer, trace_model


class FakeModel:
    """Offline stand-in for a BaseLLM provider with flat pricing."""

    PROVIDER_NAME = "FakeProvider"

    def __init__(self):
        self.model_name = "fake-model-1"
        self.cost_per_token = 0.00002

    async def generate(self, prompt, **kwargs):
        await asyncio.sleep(0.01)
        return f"Answer based on {len(prompt)} chars of context."


def fake_retriever(query):
    return [
        "MultiMind traces runs and spans to SQLite and the platform.",
        "Content is hashed by default; store_content=True opts in.",
    ]


async def answer(tracer, model, question):
    with tracer.run("qa_chain", inputs=question) as chain:
        with chain.span("retrieve", "tool", inputs=question) as retrieval:
            docs = fake_retriever(question)
            retrieval.end(outputs=docs)
        prompt = f"Context:\n{chr(10).join(docs)}\n\nQuestion: {question}"
        answer_text = await model.generate(prompt)
        chain.end(outputs=answer_text)
    return answer_text


def print_run_tree(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY start_time")]
    conn.close()
    children = {}
    for row in rows:
        children.setdefault(row["parent_id"], []).append(row)

    def render(row, depth):
        indent = "  " * depth
        tokens = json.loads(row["tokens"]) if row["tokens"] else None
        detail = f"in={row['input_chars']}ch out={row['output_chars']}ch"
        if tokens:
            detail += f" tokens={tokens['input']}+{tokens['output']}"
            detail += " (est)" if tokens["estimated"] else ""
        if row["cost"] is not None:
            detail += f" cost=${row['cost']:.6f}"
        print(f"{indent}- [{row['run_type']}] {row['name']} ({row['status']}) {detail}")
        print(
            f"{indent}    input_sha256={row['input_sha256'][:16]}... inputs_stored={row['inputs'] is not None}"
        )
        for child in children.get(row["id"], []):
            render(child, depth + 1)

    for root in children.get(None, []):
        render(root, 0)


async def main():
    db_path = Path(tempfile.mkdtemp()) / "traces.db"
    tracer = RunTracer(project="tracing-demo", sqlite_path=db_path, tags=["demo"])
    model = trace_model(FakeModel(), tracer=tracer)

    question = "How does MultiMind tracing keep prompts private?"
    answer_text = await answer(tracer, model, question)

    print(f"Question: {question}")
    print(f"Answer:   {answer_text}")
    print(f"\nStored run tree ({db_path}):\n")
    print_run_tree(db_path)


if __name__ == "__main__":
    asyncio.run(main())
