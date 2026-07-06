"""Shared guard plumbing for the framework adapters.

``GuardState`` bundles the governance pieces every adapter needs — PII
detection/redaction/blocking (:mod:`multimind.compliance.guard`), JSONL
audit, and cost/budget tracking (:mod:`multimind.observability.cost_tracker`)
— behind a text-level API so each adapter only handles its framework's
message shapes.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from multimind.compliance.guard import (
    _LABELS,
    _STRATEGIES,
    AuditLog,
    ComplianceViolationError,
    PIIDetector,
    PIIMatch,
    _hash_tag,
)
from multimind.observability.cost_tracker import (
    Budget,
    CostTracker,
    _resolve_cost,
    estimate_tokens,
)

__all__ = ["GuardState"]


class GuardState:
    """PII, audit, and cost state shared by all framework adapters.

    ``pricing`` is an optional ``{model_prefix: price}`` map (float per token,
    or ``{"input": ..., "output": ...}``) used to cost calls made through a
    foreign framework; unmatched models are recorded at $0 with
    ``unpriced=True``, never a fabricated price.
    """

    def __init__(
        self,
        framework: str,
        redact_input: bool = True,
        redact_output: bool = True,
        strategy: str = "mask",
        block_on: Tuple[str, ...] = (),
        detector: Optional[PIIDetector] = None,
        audit_log: Optional[Any] = None,
        tracker: Optional[CostTracker] = None,
        budget: Optional[Budget] = None,
        tag: Optional[str] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ):
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {strategy!r} (use one of {_STRATEGIES})")
        self.framework = framework
        self.redact_input = redact_input
        self.redact_output = redact_output
        self.strategy = strategy
        self.block_on = tuple(block_on)
        self.detector = detector or PIIDetector()
        self.tracker = tracker
        self.budget = budget
        self.tag = tag
        self.pricing = pricing
        if audit_log is None or isinstance(audit_log, AuditLog):
            self.audit_log = audit_log
        else:
            self.audit_log = AuditLog(audit_log)

    def screen_input(self, text: str, method: str) -> str:
        matches = self.detector.detect(text)
        self._check_blocked(matches, method)
        if self.redact_input:
            text, _ = self.detector.redact(text, self.strategy)
        self._audit("input", method, matches)
        return text

    def screen_output(self, text: str, method: str) -> str:
        if self.redact_output:
            text, matches = self.detector.redact(text, self.strategy)
        else:
            matches = self.detector.detect(text)
        self._audit("output", method, matches)
        return text

    def scan_output(self, text: str, method: str) -> List[PIIMatch]:
        """Detect-and-audit only; used for streamed output that has already been yielded."""
        matches = self.detector.detect(text)
        self._audit("output", method, matches)
        return matches

    def check_budget(self) -> None:
        if self.budget is not None:
            self.budget.check()

    def record_usage(
        self,
        model_name: str,
        method: str,
        input_text: str = "",
        output_text: str = "",
        usage: Optional[Tuple[int, int]] = None,
        provider: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one call's tokens/cost; no-op unless a tracker or budget is set."""
        if self.tracker is None and self.budget is None:
            return None
        if usage is not None:
            input_tokens, output_tokens = usage
            estimated = False
        else:
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            estimated = True
        shim = SimpleNamespace(MODEL_PRICING=self.pricing or {}, model_name=model_name)
        cost, unpriced = _resolve_cost(shim, input_tokens, output_tokens)
        if self.tracker is not None:
            self.tracker.record(
                provider=provider or self.framework,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                tag=self.tag,
                method=method,
                estimated=estimated,
                unpriced=unpriced,
            )
        if self.budget is not None:
            self.budget.add(cost)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "estimated": estimated,
            "unpriced": unpriced,
        }

    def audit(self, record: Dict[str, Any]) -> None:
        """Write a free-form audit record (cost/lifecycle events)."""
        if self.audit_log is not None:
            self.audit_log.write({"framework": self.framework, **record})

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
                "framework": self.framework,
                "direction": direction,
                "method": method,
                "pii_types": dict(types),
                "count": len(matches),
                "strategy": self.strategy,
                "blocked": blocked,
                "tags": sorted(tags),
            }
        )
