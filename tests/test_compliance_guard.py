"""Tests for the runtime compliance guard (multimind.compliance.guard)."""

import json
import re

import pytest

from multimind.compliance.guard import (
    AuditLog,
    ComplianceGuard,
    ComplianceViolationError,
    PIIDetector,
    guard,
)


class MockModel:
    """Minimal BaseLLM-compatible async model for guard tests."""

    def __init__(self, response="all good", chunks=None):
        self.model_name = "mock-model"
        self.response = response
        self.chunks = chunks or []
        self.last_prompt = None
        self.last_messages = None

    async def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self.response

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        return self.response

    async def generate_stream(self, prompt, **kwargs):
        self.last_prompt = prompt
        for chunk in self.chunks:
            yield chunk

    def custom_method(self):
        return "custom-result"


@pytest.fixture
def detector():
    return PIIDetector()


# --- detection: known answers -------------------------------------------------


def _types(matches):
    return [m.type for m in matches]


def test_detects_email(detector):
    matches = detector.detect("Contact john.doe@example.com for details")
    assert _types(matches) == ["email"]
    assert matches[0].text == "john.doe@example.com"


def test_detects_phone_international(detector):
    matches = detector.detect("Call me at +1 (555) 123-4567 tomorrow")
    assert "phone" in _types(matches)


def test_detects_phone_us_plain(detector):
    matches = detector.detect("call 555-867-5309 now")
    assert _types(matches) == ["phone"]
    assert matches[0].text == "555-867-5309"


def test_detects_ssn(detector):
    matches = detector.detect("SSN is 123-45-6789 on file")
    assert _types(matches) == ["ssn"]


def test_ssn_invalid_area_rejected(detector):
    assert detector.detect("number 000-45-6789 here") == []


def test_detects_credit_card_luhn_valid(detector):
    matches = detector.detect("Card: 4111 1111 1111 1111 exp 12/28")
    assert "credit_card" in _types(matches)
    assert matches[0].text == "4111 1111 1111 1111"


def test_luhn_rejects_non_card_16_digits(detector):
    # 1234567812345678 fails the Luhn checksum
    assert detector.detect("ref 1234567812345678 attached") == []


def test_detects_ip_address(detector):
    matches = detector.detect("server at 192.168.1.100 responded")
    assert _types(matches) == ["ip_address"]


def test_rejects_invalid_ip_octets(detector):
    assert detector.detect("version 999.999.999.999 string") == []


def test_detects_iban(detector):
    matches = detector.detect("IBAN DE89370400440532013000 for transfer")
    assert _types(matches) == ["iban"]


def test_detects_passport(detector):
    matches = detector.detect("Passport No X1234567 was scanned")
    assert "passport" in _types(matches)


def test_detects_dob_in_context(detector):
    matches = detector.detect("Patient DOB: 04/12/1985 admitted")
    assert _types(matches) == ["dob"]
    assert matches[0].text == "04/12/1985"


def test_detects_born_on_dob(detector):
    matches = detector.detect("She was born on January 5, 1990 in Berlin")
    assert _types(matches) == ["dob"]


def test_dob_date_without_context_ignored(detector):
    assert detector.detect("meeting on 04/12/1985 agenda") == []


def test_detects_openai_style_key(detector):
    matches = detector.detect("use key sk-abc123def456ghi789jkl here")
    assert _types(matches) == ["api_key"]


def test_detects_aws_key(detector):
    matches = detector.detect("aws AKIAIOSFODNN7EXAMPLE configured")
    assert _types(matches) == ["api_key"]


def test_detects_github_token(detector):
    token = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"
    matches = detector.detect(f"token {token} in env")
    assert _types(matches) == ["api_key"]


def test_detects_bearer_token(detector):
    matches = detector.detect("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.x.y")
    assert _types(matches) == ["api_key"]


def test_detects_high_entropy_hex(detector):
    matches = detector.detect("secret 3f7a9b2c8d1e4f6a5b3c7d9e2f4a6b8c leaked")
    assert _types(matches) == ["api_key"]


def test_entropy_rejects_low_entropy_token(detector):
    assert detector.detect("id aaaaaaaabbbbbbbbccccccccdddddddd here") == []


def test_no_detections_in_normal_prose(detector):
    text = "The quick brown fox jumps over the lazy dog every single day"
    assert detector.detect(text) == []


# --- redaction strategies -----------------------------------------------------


def test_redact_mask(detector):
    redacted, matches = detector.redact("Email john.doe@example.com now", "mask")
    assert redacted == "Email [EMAIL] now"
    assert len(matches) == 1


def test_redact_hash_is_stable(detector):
    text = "a@b.com wrote to c@d.com then a@b.com again"
    redacted, _ = detector.redact(text, "hash")
    tags = re.findall(r"\[EMAIL:([0-9a-f]{8})\]", redacted)
    assert len(tags) == 3
    assert tags[0] == tags[2]
    assert tags[0] != tags[1]


def test_redact_remove(detector):
    redacted, _ = detector.redact("Email john.doe@example.com now", "remove")
    assert "john.doe@example.com" not in redacted
    assert "[EMAIL]" not in redacted


def test_redact_unknown_strategy_raises(detector):
    with pytest.raises(ValueError):
        detector.redact("anything", "rot13")


# --- ComplianceGuard ------------------------------------------------------


async def test_guard_redacts_input():
    model = MockModel()
    guarded = ComplianceGuard(model)
    await guarded.generate("My email is john.doe@example.com ok?")
    assert model.last_prompt == "My email is [EMAIL] ok?"


async def test_guard_redacts_output():
    model = MockModel(response="the ssn is 123-45-6789 indeed")
    guarded = ComplianceGuard(model)
    result = await guarded.generate("hello")
    assert result == "the ssn is [SSN] indeed"


async def test_guard_passthrough_when_disabled():
    model = MockModel(response="ip 10.0.0.1 seen")
    guarded = ComplianceGuard(model, redact_input=False, redact_output=False)
    result = await guarded.generate("mail a@b.com")
    assert model.last_prompt == "mail a@b.com"
    assert result == "ip 10.0.0.1 seen"


async def test_block_on_raises():
    model = MockModel()
    guarded = ComplianceGuard(model, block_on=("credit_card", "ssn"))
    with pytest.raises(ComplianceViolationError) as excinfo:
        await guarded.generate("charge card 4111 1111 1111 1111 please")
    assert "credit_card" in str(excinfo.value)
    assert model.last_prompt is None  # never reached the model


async def test_chat_redacts_messages_without_mutating_original():
    model = MockModel()
    guarded = ComplianceGuard(model)
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "reach me at john.doe@example.com"},
    ]
    await guarded.chat(messages)
    assert model.last_messages[1]["content"] == "reach me at [EMAIL]"
    assert messages[1]["content"] == "reach me at john.doe@example.com"


async def test_audit_log_valid_jsonl_without_raw_pii(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    model = MockModel(response="reply for 123-45-6789 done")
    guarded = ComplianceGuard(model, audit_log=log_path, strategy="hash")
    await guarded.generate("email john.doe@example.com please")

    raw = log_path.read_text()
    assert "john.doe@example.com" not in raw
    assert "123-45-6789" not in raw

    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 2
    assert records[0]["direction"] == "input"
    assert records[0]["pii_types"] == {"email": 1}
    assert records[0]["count"] == 1
    assert records[0]["strategy"] == "hash"
    assert records[0]["blocked"] is False
    assert "timestamp" in records[0]
    assert records[1]["direction"] == "output"
    assert records[1]["pii_types"] == {"ssn": 1}


async def test_audit_log_records_blocked_event(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    guarded = ComplianceGuard(MockModel(), audit_log=str(log_path), block_on=("ssn",))
    with pytest.raises(ComplianceViolationError):
        await guarded.generate("ssn 123-45-6789")
    record = json.loads(log_path.read_text().splitlines()[0])
    assert record["blocked"] is True
    assert record["pii_types"] == {"ssn": 1}


def test_getattr_delegation():
    model = MockModel()
    guarded = ComplianceGuard(model)
    assert guarded.model_name == "mock-model"
    assert guarded.custom_method() == "custom-result"


async def test_stream_redaction_across_chunk_boundary():
    chunks = [
        "This is a long preamble sentence to fill the buffer nicely. ",
        "Contact me at john",
        ".doe@example.com for details, thanks a lot for reading this.",
    ]
    model = MockModel(chunks=chunks)
    guarded = ComplianceGuard(model)
    collected = "".join([chunk async for chunk in guarded.generate_stream("hi")])
    assert "john.doe@example.com" not in collected
    assert "[EMAIL]" in collected
    assert collected == (
        "This is a long preamble sentence to fill the buffer nicely. "
        "Contact me at [EMAIL] for details, thanks a lot for reading this."
    )


def test_guard_convenience_function():
    guarded = guard(MockModel(), strategy="hash", block_on=("ssn",))
    assert isinstance(guarded, ComplianceGuard)
    assert guarded.strategy == "hash"
    assert guarded.block_on == ("ssn",)


def test_audit_log_accepts_file_like():
    import io

    stream = io.StringIO()
    log = AuditLog(stream)
    log.write({"direction": "input", "count": 0})
    record = json.loads(stream.getvalue())
    assert record["direction"] == "input"
    assert "timestamp" in record


def test_package_exports():
    from multimind.compliance import (
        AuditLog as PkgAuditLog,
    )
    from multimind.compliance import (
        ComplianceGuard as PkgGuard,
    )
    from multimind.compliance import (
        ComplianceViolationError as PkgError,
    )
    from multimind.compliance import (
        PIIDetector as PkgDetector,
    )
    from multimind.compliance import (
        guard as pkg_guard,
    )

    assert PkgGuard is ComplianceGuard
    assert PkgDetector is PIIDetector
    assert pkg_guard is guard
    assert PkgError is ComplianceViolationError
    assert PkgAuditLog is AuditLog
