"""First-class cost tracking for MultiMind models.

``CostTracker`` accumulates per-call token/cost records; ``track_costs``
wraps any BaseLLM-compatible model (same ``__getattr__`` delegation pattern
as :class:`multimind.compliance.guard.ComplianceGuard`) and records usage
after every generate/chat call. ``Budget`` adds an optional spend ceiling.
JSONL persistence records tokens/costs/flags only — never prompt or
response content (same principle as the compliance ``AuditLog``).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised before a call when a :class:`Budget` ceiling has been reached."""

    def __init__(self, message: str, spent: float = 0.0, max_cost: float = 0.0):
        super().__init__(message)
        self.spent = spent
        self.max_cost = max_cost


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4 heuristic); an estimate, not a count."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class CostRecord:
    """One tracked call: token counts, cost, and provenance flags."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    tag: Optional[str] = None
    method: Optional[str] = None
    estimated: bool = False
    unpriced: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Budget:
    """Session spend ceiling with a soft warning threshold (default 80%)."""

    PERIODS = ("session",)

    def __init__(self, max_cost: float, period: str = "session", warn_ratio: float = 0.8):
        if max_cost <= 0:
            raise ValueError("max_cost must be positive")
        if period not in self.PERIODS:
            raise ValueError(f"Unknown budget period: {period!r} (use one of {self.PERIODS})")
        if not 0 < warn_ratio <= 1:
            raise ValueError("warn_ratio must be in (0, 1]")
        self.max_cost = max_cost
        self.period = period
        self.warn_ratio = warn_ratio
        self._spent = 0.0
        self._warned = False
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.max_cost - self._spent)

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if the ceiling has been reached."""
        with self._lock:
            if self._spent >= self.max_cost:
                raise BudgetExceededError(
                    f"Budget of ${self.max_cost:.6f} exceeded "
                    f"(spent ${self._spent:.6f}); call blocked before dispatch.",
                    spent=self._spent,
                    max_cost=self.max_cost,
                )

    def add(self, cost: float) -> None:
        with self._lock:
            self._spent += cost
            if not self._warned and self._spent >= self.max_cost * self.warn_ratio:
                self._warned = True
                logger.warning(
                    "Budget at %.0f%%: spent $%.6f of $%.6f",
                    100 * self._spent / self.max_cost,
                    self._spent,
                    self.max_cost,
                )

    def reset(self) -> None:
        with self._lock:
            self._spent = 0.0
            self._warned = False


class CostTracker:
    """Thread-safe accumulator of :class:`CostRecord` entries.

    Pass ``jsonl_path`` to append each record as one JSON line; records hold
    tokens/costs/flags only, never prompt or response content.
    """

    def __init__(self, jsonl_path: Optional[Union[str, Path]] = None):
        self._records: List[CostRecord] = []
        self._lock = threading.Lock()
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        tag: Optional[str] = None,
        method: Optional[str] = None,
        estimated: bool = False,
        unpriced: bool = False,
    ) -> CostRecord:
        entry = CostRecord(
            provider=provider,
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost=float(cost),
            tag=tag,
            method=method,
            estimated=estimated,
            unpriced=unpriced,
        )
        with self._lock:
            self._records.append(entry)
            if self._jsonl_path is not None:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_dict(), default=str) + "\n")
        return entry

    @property
    def records(self) -> List[CostRecord]:
        with self._lock:
            return list(self._records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def by_model(self) -> Dict[str, Dict[str, Any]]:
        return self._group_by(lambda r: r.model)

    def by_provider(self) -> Dict[str, Dict[str, Any]]:
        return self._group_by(lambda r: r.provider)

    def by_tag(self) -> Dict[str, Dict[str, Any]]:
        return self._group_by(lambda r: r.tag if r.tag is not None else "untagged")

    def _group_by(self, key) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            g = groups.setdefault(
                key(r),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            )
            g["calls"] += 1
            g["input_tokens"] += r.input_tokens
            g["output_tokens"] += r.output_tokens
            g["cost"] += r.cost
        return groups

    def summary(self) -> Dict[str, Any]:
        records = self.records
        return {
            "calls": len(records),
            "total_cost": sum(r.cost for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "input_tokens": sum(r.input_tokens for r in records),
            "output_tokens": sum(r.output_tokens for r in records),
            "estimated_calls": sum(1 for r in records if r.estimated),
            "unpriced_calls": sum(1 for r in records if r.unpriced),
            "by_model": self.by_model(),
            "by_provider": self.by_provider(),
            "by_tag": self.by_tag(),
        }

    def report(self) -> str:
        """Plain-text cost report table (no rich dependency)."""
        summary = self.summary()
        headers = ("model", "provider", "calls", "in_tok", "out_tok", "cost_usd")
        rows: List[Tuple[str, ...]] = []
        per_model: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            g = per_model.setdefault(
                (r.model, r.provider),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            )
            g["calls"] += 1
            g["input_tokens"] += r.input_tokens
            g["output_tokens"] += r.output_tokens
            g["cost"] += r.cost
        for (model, provider), g in sorted(per_model.items()):
            rows.append(
                (
                    model,
                    provider,
                    str(g["calls"]),
                    str(g["input_tokens"]),
                    str(g["output_tokens"]),
                    f"{g['cost']:.6f}",
                )
            )
        rows.append(
            (
                "TOTAL",
                "",
                str(summary["calls"]),
                str(summary["input_tokens"]),
                str(summary["output_tokens"]),
                f"{summary['total_cost']:.6f}",
            )
        )
        widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
        sep = "-+-".join("-" * w for w in widths)
        lines = ["Cost report", "=" * len(sep)]
        lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
        lines.append(sep)
        for row in rows[:-1]:
            lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
        lines.append(sep)
        lines.append(" | ".join(c.ljust(w) for c, w in zip(rows[-1], widths)))
        notes = []
        if summary["estimated_calls"]:
            notes.append(f"{summary['estimated_calls']} call(s) with estimated token counts")
        if summary["unpriced_calls"]:
            notes.append(f"{summary['unpriced_calls']} unpriced call(s) recorded at $0")
        if notes:
            lines.append("note: " + "; ".join(notes))
        return "\n".join(lines)

    def chargeback(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Per-tag cost attribution for chargeback reports.

        ``period`` filters records by ISO-8601 timestamp prefix (e.g.
        ``"2026-07"`` or ``"2026-07-06"``); ``None`` covers everything.
        Untagged records are grouped under ``"(untagged)"``.
        """
        records = self.records
        if period is not None:
            records = [r for r in records if r.timestamp.startswith(period)]
        by_tag: Dict[str, Dict[str, Any]] = {}
        for r in records:
            g = by_tag.setdefault(
                r.tag if r.tag is not None else "(untagged)",
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "estimated_calls": 0,
                    "unpriced_calls": 0,
                },
            )
            g["calls"] += 1
            g["input_tokens"] += r.input_tokens
            g["output_tokens"] += r.output_tokens
            g["total_tokens"] += r.total_tokens
            g["cost"] += r.cost
            g["estimated_calls"] += 1 if r.estimated else 0
            g["unpriced_calls"] += 1 if r.unpriced else 0
        total_cost = sum(g["cost"] for g in by_tag.values())
        for g in by_tag.values():
            g["share_pct"] = 100.0 * g["cost"] / total_cost if total_cost else 0.0
        return {
            "period": period,
            "calls": len(records),
            "total_cost": total_cost,
            "by_tag": by_tag,
        }

    def report_chargeback(self, period: Optional[str] = None) -> str:
        """Plain-text per-tag chargeback table (same style as :meth:`report`)."""
        data = self.chargeback(period=period)
        headers = ("tag", "calls", "tokens", "cost_usd", "share")
        rows: List[Tuple[str, ...]] = []
        by_tag = sorted(data["by_tag"].items(), key=lambda kv: kv[1]["cost"], reverse=True)
        for tag, g in by_tag:
            rows.append(
                (
                    tag,
                    str(g["calls"]),
                    str(g["total_tokens"]),
                    f"{g['cost']:.6f}",
                    f"{g['share_pct']:.1f}%",
                )
            )
        total_tokens = sum(g["total_tokens"] for _, g in by_tag)
        rows.append(
            (
                "TOTAL",
                str(data["calls"]),
                str(total_tokens),
                f"{data['total_cost']:.6f}",
                "100.0%" if data["total_cost"] else "0.0%",
            )
        )
        widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
        sep = "-+-".join("-" * w for w in widths)
        title = "Chargeback report" + (f" (period {data['period']})" if data["period"] else "")
        lines = [title, "=" * len(sep)]
        lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
        lines.append(sep)
        for row in rows[:-1]:
            lines.append(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
        lines.append(sep)
        lines.append(" | ".join(c.ljust(w) for c, w in zip(rows[-1], widths)))
        notes = []
        estimated = sum(g["estimated_calls"] for _, g in by_tag)
        unpriced = sum(g["unpriced_calls"] for _, g in by_tag)
        if estimated:
            notes.append(f"{estimated} call(s) with estimated token counts")
        if unpriced:
            notes.append(f"{unpriced} unpriced call(s) recorded at $0")
        if notes:
            lines.append("note: " + "; ".join(notes))
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


def load_tracker(jsonl_path: Union[str, Path]) -> CostTracker:
    """Rebuild a :class:`CostTracker` from a persisted JSONL cost log.

    The returned tracker is detached from the file (reading it does not
    re-append), so chargeback/cost reports can be run offline over
    historical logs. Original timestamps are preserved.
    """
    tracker = CostTracker()
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            record = CostRecord(
                provider=entry.get("provider", "unknown"),
                model=entry.get("model", "unknown"),
                input_tokens=int(entry.get("input_tokens", 0)),
                output_tokens=int(entry.get("output_tokens", 0)),
                cost=float(entry.get("cost", 0.0)),
                tag=entry.get("tag"),
                method=entry.get("method"),
                estimated=bool(entry.get("estimated", False)),
                unpriced=bool(entry.get("unpriced", False)),
            )
            if entry.get("timestamp"):
                record.timestamp = str(entry["timestamp"])
            tracker._records.append(record)
    return tracker


def _resolve_cost(model: Any, input_tokens: int, output_tokens: int) -> Tuple[float, bool]:
    """Resolve call cost from the wrapped model; ``(cost, unpriced_flag)``.

    Order: instance ``cost_per_token`` (already encodes provider pricing
    lookups and explicit overrides), then class ``MODEL_PRICING`` longest-
    prefix match (float, or dict with input/output prices), else $0 with
    ``unpriced=True`` — never a fabricated price.
    """
    cost_per_token = getattr(model, "cost_per_token", None)
    if isinstance(cost_per_token, (int, float)):
        return (input_tokens + output_tokens) * float(cost_per_token), False
    pricing = getattr(model, "MODEL_PRICING", None)
    model_name = getattr(model, "model_name", "")
    if isinstance(pricing, dict) and model_name:
        for prefix in sorted(pricing, key=len, reverse=True):
            if model_name.startswith(prefix):
                price = pricing[prefix]
                if isinstance(price, dict):
                    return (
                        input_tokens * float(price.get("input", 0.0))
                        + output_tokens * float(price.get("output", 0.0)),
                        False,
                    )
                return (input_tokens + output_tokens) * float(price), False
    return 0.0, True


def _usage_tokens(usage: Any) -> Optional[Tuple[int, int]]:
    """Extract ``(input_tokens, output_tokens)`` from a usage dict/object."""
    if usage is None:
        return None

    def _get(*names: str) -> Optional[int]:
        for name in names:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    input_tokens = _get("input_tokens", "prompt_tokens")
    output_tokens = _get("output_tokens", "completion_tokens")
    if input_tokens is None and output_tokens is None:
        return None
    return input_tokens or 0, output_tokens or 0


def _extract_usage(result: Any, model: Any) -> Optional[Tuple[int, int]]:
    """Use real usage info when the call or model exposes it, else None."""
    tokens = _usage_tokens(getattr(result, "usage", None))
    if tokens is not None:
        return tokens
    get_last_usage = getattr(model, "get_last_usage", None)
    if callable(get_last_usage):
        tokens = _usage_tokens(get_last_usage())
        if tokens is not None:
            return tokens
    return _usage_tokens(getattr(model, "last_usage", None))


def _messages_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))


class TrackedModel:
    """Drop-in cost-tracking wrapper around any BaseLLM-compatible model.

    Intercepts async ``generate``/``chat``/``generate_stream``/``chat_stream``
    and sync ``generate_sync``/``chat_sync``; every other attribute is proxied
    to the wrapped model. Token counts come from the model's own usage info
    when exposed, otherwise from the chars/4 :func:`estimate_tokens` heuristic
    (recorded with ``estimated=True``).
    """

    def __init__(
        self,
        model: Any,
        tracker: Optional[CostTracker] = None,
        budget: Optional[Budget] = None,
        tag: Optional[str] = None,
    ):
        self._model = model
        self.tracker = tracker if tracker is not None else get_default_tracker()
        self.budget = budget
        self.tag = tag

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _precheck(self) -> None:
        if self.budget is not None:
            self.budget.check()

    def _record(self, method: str, input_text: str, output_text: str, result: Any = None) -> None:
        usage = _extract_usage(result, self._model)
        if usage is not None:
            input_tokens, output_tokens = usage
            estimated = False
        else:
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            estimated = True
        cost, unpriced = _resolve_cost(self._model, input_tokens, output_tokens)
        self.tracker.record(
            provider=getattr(self._model, "PROVIDER_NAME", type(self._model).__name__),
            model=getattr(self._model, "model_name", "unknown"),
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

    @staticmethod
    def _output_text(result: Any) -> str:
        return result if isinstance(result, str) else str(result)

    async def generate(self, prompt: str, **kwargs) -> Any:
        self._precheck()
        result = await self._model.generate(prompt, **kwargs)
        self._record("generate", prompt, self._output_text(result), result)
        return result

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        self._precheck()
        result = await self._model.chat(messages, **kwargs)
        self._record("chat", _messages_text(messages), self._output_text(result), result)
        return result

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        self._precheck()
        chunks: List[str] = []
        async for chunk in self._model.generate_stream(prompt, **kwargs):
            chunks.append(chunk)
            yield chunk
        self._record("generate_stream", prompt, "".join(chunks))

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        self._precheck()
        chunks: List[str] = []
        async for chunk in self._model.chat_stream(messages, **kwargs):
            chunks.append(chunk)
            yield chunk
        self._record("chat_stream", _messages_text(messages), "".join(chunks))

    def generate_sync(self, prompt: str, **kwargs) -> Any:
        self._precheck()
        result = self._model.generate_sync(prompt, **kwargs)
        self._record("generate_sync", prompt, self._output_text(result), result)
        return result

    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        self._precheck()
        result = self._model.chat_sync(messages, **kwargs)
        self._record("chat_sync", _messages_text(messages), self._output_text(result), result)
        return result


_default_tracker: Optional[CostTracker] = None
_default_tracker_lock = threading.Lock()


def get_default_tracker() -> CostTracker:
    """Return the process-wide session tracker, creating it on first use."""
    global _default_tracker
    with _default_tracker_lock:
        if _default_tracker is None:
            _default_tracker = CostTracker()
        return _default_tracker


def reset_default_tracker() -> None:
    global _default_tracker
    with _default_tracker_lock:
        _default_tracker = None


def track_costs(
    model: Any,
    tracker: Optional[CostTracker] = None,
    budget: Optional[Budget] = None,
    tag: Optional[str] = None,
) -> TrackedModel:
    """Convenience wrapper: ``track_costs(model) -> TrackedModel`` on the default tracker."""
    return TrackedModel(model, tracker=tracker, budget=budget, tag=tag)


def cost_summary(tracker: Optional[CostTracker] = None) -> Dict[str, Any]:
    """Print the session cost report and return the summary dict."""
    tracker = tracker if tracker is not None else get_default_tracker()
    print(tracker.report())
    return tracker.summary()
