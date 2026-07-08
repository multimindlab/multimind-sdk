"""Runtime compliance guard: drop-in PII detection/redaction middleware.

Wraps any BaseLLM-compatible model (duck-typed async ``generate`` / ``chat`` /
``generate_stream``) with regex-based PII detection, redaction, blocking, and
a JSONL audit trail. Stdlib-only (``re``, ``hashlib``, ``math``); optionally
augmented by ``presidio_analyzer`` when installed and enabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ComplianceViolationError(Exception):
    """Raised when a ``block_on`` PII type is found in guarded input."""

    def __init__(self, message: str, matches: Optional[List["PIIMatch"]] = None):
        super().__init__(message)
        self.matches = matches or []


@dataclass
class PIIMatch:
    """A single PII detection: type, character span, and matched text."""

    type: str
    start: int
    end: int
    text: str


_LABELS = {
    "email": "EMAIL",
    "phone": "PHONE",
    "ssn": "SSN",
    "credit_card": "CREDIT_CARD",
    "ip_address": "IP_ADDRESS",
    "iban": "IBAN",
    "passport": "PASSPORT",
    "dob": "DOB",
    "api_key": "API_KEY",
}

_STRATEGIES = ("mask", "hash", "remove")


def _hash_tag(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _valid_card(text: str) -> bool:
    digits = re.sub(r"[ -]", "", text)
    return digits.isdigit() and 13 <= len(digits) <= 19 and _luhn_valid(digits)


def _valid_ssn(text: str) -> bool:
    area, group, serial = text.split("-")
    if area in ("000", "666") or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def _valid_phone(text: str) -> bool:
    return 7 <= len(re.sub(r"\D", "", text)) <= 15


def _valid_ip(text: str) -> bool:
    return all(0 <= int(part) <= 255 for part in text.split("."))


def _min_entropy(threshold: float) -> Callable[[str], bool]:
    return lambda text: _shannon_entropy(text) >= threshold


# (type, pattern, validator, capture-group). List order doubles as the
# priority used to resolve overlapping matches.
_PATTERNS: List[Tuple[str, "re.Pattern[str]", Optional[Callable[[str], bool]], int]] = [
    ("credit_card", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), _valid_card, 0),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), None, 0),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), _valid_ssn, 0),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        None,
        0,
    ),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), None, 0),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), None, 0),
    ("api_key", re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), None, 0),
    ("api_key", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{16,})"), None, 1),
    ("api_key", re.compile(r"\b[0-9a-fA-F]{32,}\b"), _min_entropy(3.0), 0),
    ("api_key", re.compile(r"\b[A-Za-z0-9+/_=-]{32,}\b"), _min_entropy(4.0), 0),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _valid_ip, 0),
    (
        "dob",
        re.compile(
            r"(?i)\b(?:dob|date of birth|born on)\b\s*[:=]?\s*"
            r"(\d{4}-\d{2}-\d{2}"
            r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
            r"|[A-Za-z]+ \d{1,2},? \d{4})"
        ),
        None,
        1,
    ),
    (
        "phone",
        re.compile(
            r"(?:\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){1,4})"
            r"|(?:\(\d{2,4}\)[ .-]?\d{3,4}[ .-]?\d{3,4})"
            r"|(?:\b\d{3}[-.]\d{3}[-.]\d{4}\b)"
        ),
        _valid_phone,
        0,
    ),
    ("passport", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), None, 0),
]


class PIIDetector:
    """Regex-based PII detector with Luhn and Shannon-entropy false-positive filters.

    Set ``use_presidio=True`` to augment detection with ``presidio_analyzer``
    when it is installed (silently skipped otherwise).
    """

    def __init__(self, use_presidio: bool = False):
        self._analyzer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine

                self._analyzer = AnalyzerEngine()
            except ImportError:
                self._analyzer = None

    def detect(self, text: str) -> List[PIIMatch]:
        """Return non-overlapping PII matches sorted by position."""
        candidates: List[Tuple[int, int, int, PIIMatch]] = []
        for priority, (ptype, pattern, validator, group) in enumerate(_PATTERNS):
            for m in pattern.finditer(text):
                value = m.group(group)
                if value is None:
                    continue
                if validator is not None and not validator(value):
                    continue
                start, end = m.span(group)
                candidates.append(
                    (start, start - end, priority, PIIMatch(ptype, start, end, value))
                )
        if self._analyzer is not None:
            base = len(_PATTERNS)
            for i, res in enumerate(self._analyzer.analyze(text=text, language="en")):
                match = PIIMatch(
                    res.entity_type.lower(), res.start, res.end, text[res.start : res.end]
                )
                candidates.append((res.start, res.start - res.end, base + i, match))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        accepted: List[PIIMatch] = []
        last_end = 0
        for start, _, _, match in candidates:
            if start < last_end:
                continue
            accepted.append(match)
            last_end = match.end
        return accepted

    def redact(self, text: str, strategy: str = "mask") -> Tuple[str, List[PIIMatch]]:
        """Redact detected PII, returning ``(redacted_text, matches)``."""
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {strategy!r} (use one of {_STRATEGIES})")
        matches = self.detect(text)
        out = text
        for m in reversed(matches):
            out = out[: m.start] + self._replacement(m, strategy) + out[m.end :]
        return out, matches

    @staticmethod
    def _replacement(match: PIIMatch, strategy: str) -> str:
        label = _LABELS.get(match.type, match.type.upper())
        if strategy == "mask":
            return f"[{label}]"
        if strategy == "hash":
            return f"[{label}:{_hash_tag(match.text)}]"
        return ""


class AuditLog:
    """Append-only JSONL audit trail; records types/counts/hash tags, never raw PII."""

    def __init__(self, destination: Union[str, Path, Any]):
        if isinstance(destination, (str, Path)):
            self._path: Optional[Path] = Path(destination)
            self._stream = None
        else:
            self._path = None
            self._stream = destination

    def write(self, record: Dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(entry, default=str) + "\n"
        if self._stream is not None:
            self._stream.write(line)
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line)


class ComplianceGuard:
    """Drop-in PII guard around any BaseLLM-compatible model.

    Redacts PII from prompts before they reach the model, optionally scans and
    redacts model output, raises :class:`ComplianceViolationError` when a
    ``block_on`` PII type appears in input, and writes every event to an
    optional :class:`AuditLog`. All other attribute access is proxied to the
    wrapped model, so the guard is a drop-in replacement.

    Streaming: ``generate_stream`` holds back a small overlap window across
    chunk boundaries before yielding, trading a little latency for detection
    of PII split across chunks.
    """

    def __init__(
        self,
        model: Any,
        redact_input: bool = True,
        redact_output: bool = True,
        strategy: str = "mask",
        audit_log: Optional[Union[AuditLog, str, Path, Any]] = None,
        block_on: Tuple[str, ...] = (),
        detector: Optional[PIIDetector] = None,
        stream_overlap: int = 64,
    ):
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {strategy!r} (use one of {_STRATEGIES})")
        self._model = model
        self.redact_input = redact_input
        self.redact_output = redact_output
        self.strategy = strategy
        self.block_on = tuple(block_on)
        self.detector = detector or PIIDetector()
        self.stream_overlap = stream_overlap
        if audit_log is None or isinstance(audit_log, AuditLog):
            self.audit_log = audit_log
        else:
            self.audit_log = AuditLog(audit_log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    async def generate(self, prompt: str, **kwargs) -> str:
        prompt = self._screen_input(prompt, "generate")
        output = await self._model.generate(prompt, **kwargs)
        return self._screen_output(output, "generate")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        all_matches: List[PIIMatch] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                all_matches.extend(self.detector.detect(content))
        self._check_blocked(all_matches, "chat")
        if self.redact_input:
            messages = [
                {**msg, "content": self.detector.redact(msg["content"], self.strategy)[0]}
                if isinstance(msg.get("content"), str)
                else msg
                for msg in messages
            ]
        self._audit("input", "chat", all_matches)
        output = await self._model.chat(messages, **kwargs)
        return self._screen_output(output, "chat")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        prompt = self._screen_input(prompt, "generate_stream")
        buffer = ""
        out_matches: List[PIIMatch] = []
        async for chunk in self._model.generate_stream(prompt, **kwargs):
            buffer += chunk
            if len(buffer) <= self.stream_overlap:
                continue
            emit_upto = len(buffer) - self.stream_overlap
            for m in self.detector.detect(buffer):
                if m.start < emit_upto < m.end:
                    emit_upto = m.start
            if emit_upto <= 0:
                continue
            segment, buffer = buffer[:emit_upto], buffer[emit_upto:]
            yield self._emit(segment, out_matches)
        if buffer:
            yield self._emit(buffer, out_matches)
        self._audit("output", "generate_stream", out_matches)

    def _emit(self, segment: str, out_matches: List[PIIMatch]) -> str:
        if self.redact_output:
            segment, matches = self.detector.redact(segment, self.strategy)
        else:
            matches = self.detector.detect(segment)
        out_matches.extend(matches)
        return segment

    def _screen_input(self, text: str, method: str) -> str:
        matches = self.detector.detect(text)
        self._check_blocked(matches, method)
        if self.redact_input:
            text, _ = self.detector.redact(text, self.strategy)
        self._audit("input", method, matches)
        return text

    def _screen_output(self, text: str, method: str) -> str:
        if self.redact_output:
            text, matches = self.detector.redact(text, self.strategy)
        else:
            matches = self.detector.detect(text)
        self._audit("output", method, matches)
        return text

    def _check_blocked(self, matches: List[PIIMatch], method: str) -> None:
        blocked = sorted({m.type for m in matches if m.type in self.block_on})
        if blocked:
            self._audit("input", method, matches, blocked=True)
            raise ComplianceViolationError(
                f"Blocked PII types found in input: {', '.join(blocked)}", matches
            )

    def _audit(
        self, direction: str, method: str, matches: List[PIIMatch], blocked: bool = False
    ) -> None:
        if self.audit_log is None:
            return
        types = Counter(m.type for m in matches)
        tags = {f"[{_LABELS.get(m.type, m.type.upper())}:{_hash_tag(m.text)}]" for m in matches}
        self.audit_log.write(
            {
                "direction": direction,
                "method": method,
                "pii_types": dict(types),
                "count": len(matches),
                "strategy": self.strategy,
                "blocked": blocked,
                "tags": sorted(tags),
            }
        )


def guard(model: Any, **kwargs) -> ComplianceGuard:
    """Convenience wrapper: ``guard(model, block_on=("ssn",)) -> ComplianceGuard``."""
    return ComplianceGuard(model, **kwargs)
