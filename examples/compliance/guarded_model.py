"""
Runtime compliance guard demo: PII detection, redaction, blocking, and audit trail.

Uses a tiny in-file fake model, so it runs end-to-end with no API keys:
    python examples/compliance/guarded_model.py
"""

import asyncio
import tempfile
from pathlib import Path

from multimind.compliance.guard import (
    ComplianceGuard,
    ComplianceViolationError,
    PIIDetector,
)


class EchoModel:
    """Fake model that echoes what it received (so redaction is visible)."""

    model_name = "echo-fake-model"

    async def generate(self, prompt, **kwargs):
        return f"Model received: {prompt}"

    async def chat(self, messages, **kwargs):
        return f"Model received: {messages[-1]['content']}"

    async def generate_stream(self, prompt, **kwargs):
        response = "Sure - the customer's card is 4111 1111 1111 1111, noted."
        for i in range(0, len(response), 12):
            yield response[i : i + 12]


async def main():
    text = (
        "Hi, I'm Jane (DOB: 04/12/1985). Email jane.doe@example.com, "
        "phone +1 (555) 123-4567, SSN 123-45-6789, api key sk-abc123def456ghi789jkl."
    )

    print("=== 1. Detection ===")
    detector = PIIDetector()
    for match in detector.detect(text):
        print(f"  {match.type:<10} span=({match.start},{match.end}) text={match.text!r}")

    print("\n=== 2. Redaction strategies ===")
    for strategy in ("mask", "hash", "remove"):
        redacted, _ = detector.redact(text, strategy)
        print(f"  [{strategy}] {redacted}")

    audit_path = Path(tempfile.mkdtemp()) / "audit.jsonl"
    guarded = ComplianceGuard(
        EchoModel(),
        strategy="mask",
        audit_log=audit_path,
        block_on=("credit_card",),
    )

    print("\n=== 3. Guarded generate (input redacted before the model sees it) ===")
    print(" ", await guarded.generate(text))

    print("\n=== 4. Blocking ===")
    try:
        await guarded.generate("Please charge card 4111 1111 1111 1111 now")
    except ComplianceViolationError as exc:
        print(f"  Blocked: {exc}")

    print("\n=== 5. Streaming (model output redacted across chunk boundaries) ===")
    chunks = [chunk async for chunk in guarded.generate_stream("what card is on file?")]
    print(" ", "".join(chunks))

    print("\n=== 6. Audit trail (types/counts/hash tags only, never raw PII) ===")
    for line in audit_path.read_text().splitlines():
        print(" ", line)


if __name__ == "__main__":
    asyncio.run(main())
