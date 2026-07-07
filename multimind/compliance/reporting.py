"""Compliance evidence reporting: auditor-shaped documents from SDK artifacts.

``build_evidence_report`` parses whichever JSONL artifacts are provided — the
:class:`multimind.compliance.guard.AuditLog` trail, the
:class:`multimind.observability.cost_tracker.CostTracker` cost log, and an
AI inventory report — into an :class:`EvidenceReport` with deterministic
Markdown / HTML / dict renderers. The report documents only what the
artifacts actually record; it never claims certification or legal
compliance. Stdlib-only.
"""

from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

DISCLAIMER = (
    "This is technical evidence, not a legal compliance determination. "
    "It reports only what the provided artifacts record; it does not certify "
    "compliance with any law, regulation, or framework, and an absent artifact "
    "means absent evidence, not absent risk. Framework references indicate "
    "control themes the evidence supports, never controls it satisfies."
)

_MAPPING_NOTE = (
    "Each row lists control themes the artifact supports as technical evidence; "
    "no row asserts that a control is met."
)

_GAP_NOTE = (
    "Gaps are periods with no audit records; they may reflect no traffic "
    "rather than missing logging."
)

_UNTAGGED = "(untagged)"


def _read_jsonl(path: Union[str, Path]) -> Tuple[List[Dict[str, Any]], int]:
    """Parse a JSONL file; returns ``(records, unparsed_line_count)``."""
    records: List[Dict[str, Any]] = []
    unparsed = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                unparsed += 1
                continue
            if isinstance(entry, dict):
                records.append(entry)
            else:
                unparsed += 1
    return records, unparsed


def _in_period(record: Dict[str, Any], period: Optional[str]) -> bool:
    return period is None or str(record.get("timestamp", "")).startswith(period)


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _counted(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def _data_protection_section(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    pii_by_type: Dict[str, int] = {}
    blocked_types: set = set()
    blocked = 0
    for r in records:
        for ptype, n in (r.get("pii_types") or {}).items():
            pii_by_type[ptype] = pii_by_type.get(ptype, 0) + int(n)
        if r.get("blocked"):
            blocked += 1
            blocked_types.update((r.get("pii_types") or {}).keys())
    return {
        "records": len(records),
        "records_with_pii": sum(1 for r in records if r.get("count", 0)),
        "pii_events_by_type": dict(sorted(pii_by_type.items())),
        "redaction_strategy_distribution": _counted(
            r["strategy"] for r in records if r.get("strategy")
        ),
        "blocked_requests": blocked,
        "blocked_pii_types": sorted(blocked_types),
    }


def _scanned_share(
    audit_records: List[Dict[str, Any]], cost_records: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    inputs = sum(1 for r in audit_records if r.get("direction") == "input")
    if cost_records is None:
        return {
            "scanned_input_records": inputs,
            "tracked_calls": None,
            "share_pct": None,
            "note": ("not computable: costs log not provided, so total call volume is unknown"),
        }
    total = len(cost_records)
    return {
        "scanned_input_records": inputs,
        "tracked_calls": total,
        "share_pct": round(100.0 * inputs / total, 1) if total else None,
        "note": "approximate: assumes the audit log and costs log cover the same calls",
    }


def _oversight_section(records: List[Dict[str, Any]], gap_threshold_hours: float) -> Dict[str, Any]:
    stamps = sorted(ts for ts in (_parse_ts(r.get("timestamp")) for r in records) if ts)
    gaps: List[Dict[str, Any]] = []
    for prev, curr in zip(stamps, stamps[1:]):
        hours = (curr - prev).total_seconds() / 3600.0
        if hours > gap_threshold_hours:
            gaps.append(
                {
                    "from": prev.isoformat(),
                    "to": curr.isoformat(),
                    "hours": round(hours, 1),
                }
            )
    return {
        "records": len(records),
        "first_record": stamps[0].isoformat() if stamps else None,
        "last_record": stamps[-1].isoformat() if stamps else None,
        "by_direction": _counted(str(r.get("direction", "unknown")) for r in records),
        "by_method": _counted(str(r.get("method", "unknown")) for r in records),
        "gap_threshold_hours": gap_threshold_hours,
        "gaps_over_threshold": gaps,
        "records_without_timestamp": len(records) - len(stamps),
        "note": _GAP_NOTE,
    }


def _cost_governance_section(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _group(key_fn) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for r in records:
            g = groups.setdefault(key_fn(r), {"calls": 0, "cost": 0.0})
            g["calls"] += 1
            g["cost"] += float(r.get("cost", 0.0))
        return dict(sorted(groups.items()))

    budget_blocks = sum(1 for r in records if r.get("budget_blocked") or r.get("blocked"))
    return {
        "calls": len(records),
        "total_cost": sum(float(r.get("cost", 0.0)) for r in records),
        "spend_by_tag": _group(lambda r: str(r.get("tag") or _UNTAGGED)),
        "spend_by_model": _group(lambda r: str(r.get("model", "unknown"))),
        "estimated_calls": sum(1 for r in records if r.get("estimated")),
        "unpriced_calls": sum(1 for r in records if r.get("unpriced")),
        "budget_block_events": budget_blocks,
        "budget_block_note": (
            None if budget_blocks else "no budget-block events are present in the costs log"
        ),
    }


def _inventory_section(inventory: Dict[str, Any]) -> Dict[str, Any]:
    summary = inventory.get("summary") or {}
    risks = inventory.get("risks") or {}
    return {
        "root": inventory.get("root"),
        "findings": summary.get("findings", len(inventory.get("findings") or [])),
        "providers": sorted(summary.get("by_provider") or {}),
        "external_data_flow_providers": sorted(risks.get("external_data_flow_providers") or []),
        "hardcoded_key_findings": len(risks.get("hardcoded_keys") or []),
        "scanned_files": summary.get("scanned_files"),
        "skipped_files": summary.get("skipped_files"),
        "note": inventory.get("note"),
    }


def _framework_mapping(
    has_audit: bool, has_blocked: bool, has_costs: bool, has_inventory: bool
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if has_audit:
        rows.append(
            {
                "evidence": "PII detection/redaction audit trail (per-call JSONL records)",
                "source": "audit log",
                "supports": [
                    "EU AI Act Art. 12 record-keeping",
                    "SOC 2 monitoring",
                    "HIPAA 164.312(b) audit controls",
                ],
            }
        )
    if has_blocked:
        rows.append(
            {
                "evidence": "Blocked-request events for configured PII types",
                "source": "audit log",
                "supports": [
                    "SOC 2 logical access",
                    "HIPAA 164.312(b) audit controls",
                ],
            }
        )
    if has_costs:
        rows.append(
            {
                "evidence": "Per-call token/cost usage records with tags",
                "source": "costs log",
                "supports": [
                    "EU AI Act Art. 12 record-keeping",
                    "SOC 2 monitoring",
                ],
            }
        )
    if has_inventory:
        rows.append(
            {
                "evidence": "AI asset inventory (providers, data flows, key findings)",
                "source": "inventory scan",
                "supports": [
                    "EU AI Act Art. 50 transparency",
                    "SOC 2 monitoring",
                ],
            }
        )
    return rows


@dataclass
class EvidenceReport:
    """Evidence document assembled by :func:`build_evidence_report`."""

    generated_at: str
    organization: Optional[str] = None
    period: Optional[str] = None
    sources: Dict[str, Optional[str]] = field(default_factory=dict)
    missing_sources: List[str] = field(default_factory=list)
    data_protection: Optional[Dict[str, Any]] = None
    oversight: Optional[Dict[str, Any]] = None
    cost_governance: Optional[Dict[str, Any]] = None
    ai_inventory: Optional[Dict[str, Any]] = None
    framework_mapping: List[Dict[str, Any]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "organization": self.organization,
            "period": self.period,
            "sources": dict(sorted(self.sources.items())),
            "missing_sources": list(self.missing_sources),
            "data_protection": self.data_protection,
            "oversight": self.oversight,
            "cost_governance": self.cost_governance,
            "ai_inventory": self.ai_inventory,
            "framework_mapping": list(self.framework_mapping),
            "disclaimer": self.disclaimer,
        }

    # ---------------------------------------------------------------- shared

    def _meta_rows(self) -> List[Tuple[str, str]]:
        rows = [("Generated at", self.generated_at)]
        if self.organization:
            rows.append(("Organization", self.organization))
        rows.append(("Period", self.period or "all records"))
        for name in sorted(self.sources):
            rows.append((f"Source: {name}", self.sources[name] or "not provided"))
        return rows

    # -------------------------------------------------------------- markdown

    def to_markdown(self) -> str:
        lines: List[str] = ["# AI Compliance Evidence Report", ""]
        lines.append(f"> {self.disclaimer}")
        lines.append("")
        for label, value in self._meta_rows():
            lines.append(f"- **{label}:** {value}")
        lines.append("")

        lines.append("## Sources")
        lines.append("")
        for name in sorted(self.sources):
            lines.append(f"- {name}: {self.sources[name] or 'not provided'}")
        for note in self.missing_sources:
            lines.append(f"- {note}")
        lines.append("")

        lines.append("## Data protection (PII controls)")
        lines.append("")
        dp = self.data_protection
        if dp is None:
            lines.append("Audit log not provided; no data protection evidence available.")
        else:
            lines.append(f"- Audit records in scope: {dp['records']}")
            lines.append(f"- Records with PII detected: {dp['records_with_pii']}")
            lines.append(f"- Blocked requests: {dp['blocked_requests']}")
            if dp["blocked_pii_types"]:
                lines.append(f"- Blocked PII types: {', '.join(dp['blocked_pii_types'])}")
            share = dp.get("scanned_call_share") or {}
            if share.get("share_pct") is not None:
                lines.append(
                    f"- Share of tracked calls scanned: {share['share_pct']}% "
                    f"({share['scanned_input_records']} scanned inputs of "
                    f"{share['tracked_calls']} tracked calls; {share['note']})"
                )
            elif share:
                lines.append(f"- Share of tracked calls scanned: {share['note']}")
            lines.append("")
            lines.extend(
                _md_table(
                    ("PII type", "Events"),
                    sorted(dp["pii_events_by_type"].items()),
                    empty="No PII events recorded in the period.",
                )
            )
            lines.append("")
            lines.extend(
                _md_table(
                    ("Redaction strategy", "Records"),
                    sorted(dp["redaction_strategy_distribution"].items()),
                    empty="No redaction strategy recorded.",
                )
            )
        lines.append("")

        lines.append("## Oversight and audit-trail continuity")
        lines.append("")
        ov = self.oversight
        if ov is None:
            lines.append("Audit log not provided; no oversight evidence available.")
        else:
            lines.append(f"- Records in scope: {ov['records']}")
            lines.append(f"- First record: {ov['first_record'] or 'n/a'}")
            lines.append(f"- Last record: {ov['last_record'] or 'n/a'}")
            lines.append(f"- Records by direction: {_kv_inline(ov['by_direction']) or 'none'}")
            lines.append(f"- Records by method: {_kv_inline(ov['by_method']) or 'none'}")
            gaps = ov["gaps_over_threshold"]
            lines.append(f"- Gaps over {ov['gap_threshold_hours']}h threshold: {len(gaps)}")
            for gap in gaps:
                lines.append(f"  - {gap['from']} to {gap['to']} ({gap['hours']}h)")
            if ov["records_without_timestamp"]:
                lines.append(
                    f"- Records without a parseable timestamp: {ov['records_without_timestamp']}"
                )
            lines.append(f"- Note: {ov['note']}")
        lines.append("")

        lines.append("## Cost governance")
        lines.append("")
        cg = self.cost_governance
        if cg is None:
            lines.append("Costs log not provided; no cost governance evidence available.")
        else:
            lines.append(f"- Tracked calls in scope: {cg['calls']}")
            lines.append(f"- Total spend: ${cg['total_cost']:.6f}")
            lines.append(f"- Budget-block events: {cg['budget_block_events']}")
            if cg["budget_block_note"]:
                lines.append(f"- Note: {cg['budget_block_note']}")
            if cg["estimated_calls"]:
                lines.append(f"- Calls with estimated token counts: {cg['estimated_calls']}")
            if cg["unpriced_calls"]:
                lines.append(f"- Unpriced calls recorded at $0: {cg['unpriced_calls']}")
            lines.append("")
            lines.extend(
                _md_table(
                    ("Tag", "Calls", "Cost (USD)"),
                    [
                        (tag, g["calls"], f"{g['cost']:.6f}")
                        for tag, g in cg["spend_by_tag"].items()
                    ],
                    empty="No cost records in the period.",
                )
            )
            lines.append("")
            lines.extend(
                _md_table(
                    ("Model", "Calls", "Cost (USD)"),
                    [
                        (model, g["calls"], f"{g['cost']:.6f}")
                        for model, g in cg["spend_by_model"].items()
                    ],
                    empty="No cost records in the period.",
                )
            )
        lines.append("")

        lines.append("## AI asset inventory")
        lines.append("")
        inv = self.ai_inventory
        if inv is None:
            lines.append("Inventory scan not provided; no AI asset inventory evidence available.")
        else:
            lines.append(f"- Scan root: {inv['root'] or 'n/a'}")
            lines.append(f"- Findings: {inv['findings']}")
            lines.append(f"- Providers in use: {', '.join(inv['providers']) or 'none'}")
            lines.append(
                f"- External data-flow providers: "
                f"{', '.join(inv['external_data_flow_providers']) or 'none'}"
            )
            lines.append(f"- Hardcoded-key findings: {inv['hardcoded_key_findings']}")
            if inv["scanned_files"] is not None:
                lines.append(f"- Files scanned: {inv['scanned_files']}")
            if inv["note"]:
                lines.append(f"- Note: {inv['note']}")
        lines.append("")

        lines.append("## Framework control-theme mapping")
        lines.append("")
        if self.framework_mapping:
            lines.extend(
                _md_table(
                    ("Evidence", "Source", "Control themes supported"),
                    [
                        (row["evidence"], row["source"], "; ".join(row["supports"]))
                        for row in self.framework_mapping
                    ],
                )
            )
            lines.append("")
            lines.append(_MAPPING_NOTE)
        else:
            lines.append("No evidence sources provided; nothing to map.")
        lines.append("")

        lines.append("## Disclaimer")
        lines.append("")
        lines.append(self.disclaimer)
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ html

    def to_html(self) -> str:
        e = _html.escape
        parts: List[str] = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>AI Compliance Evidence Report</title>",
            "<style>",
            "body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;"
            "max-width:52rem;margin:2rem auto;padding:0 1.25rem;line-height:1.5;}",
            "h1{font-size:1.6rem;border-bottom:2px solid #1a1a1a;padding-bottom:.4rem;}",
            "h2{font-size:1.15rem;margin-top:2rem;border-bottom:1px solid #999;"
            "padding-bottom:.2rem;}",
            "table{border-collapse:collapse;width:100%;margin:.75rem 0;font-size:.95rem;}",
            "th,td{border:1px solid #999;padding:.35rem .6rem;text-align:left;vertical-align:top;}",
            "th{background:#efefef;}",
            ".disclaimer{border:1px solid #1a1a1a;background:#f7f7f7;"
            "padding:.75rem 1rem;font-style:italic;margin:1rem 0;}",
            ".note{font-size:.9rem;color:#444;}",
            "@media print{body{margin:0;max-width:none;}"
            "h2{page-break-after:avoid;}table{page-break-inside:avoid;}}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>AI Compliance Evidence Report</h1>",
            f'<p class="disclaimer">{e(self.disclaimer)}</p>',
            "<ul>",
        ]
        for label, value in self._meta_rows():
            parts.append(f"<li><strong>{e(label)}:</strong> {e(str(value))}</li>")
        parts.append("</ul>")

        parts.append("<h2>Sources</h2><ul>")
        for name in sorted(self.sources):
            parts.append(f"<li>{e(name)}: {e(self.sources[name] or 'not provided')}</li>")
        for note in self.missing_sources:
            parts.append(f"<li>{e(note)}</li>")
        parts.append("</ul>")

        parts.append("<h2>Data protection (PII controls)</h2>")
        dp = self.data_protection
        if dp is None:
            parts.append("<p>Audit log not provided; no data protection evidence available.</p>")
        else:
            items = [
                f"Audit records in scope: {dp['records']}",
                f"Records with PII detected: {dp['records_with_pii']}",
                f"Blocked requests: {dp['blocked_requests']}",
            ]
            if dp["blocked_pii_types"]:
                items.append(f"Blocked PII types: {', '.join(dp['blocked_pii_types'])}")
            share = dp.get("scanned_call_share") or {}
            if share.get("share_pct") is not None:
                items.append(
                    f"Share of tracked calls scanned: {share['share_pct']}% "
                    f"({share['scanned_input_records']} scanned inputs of "
                    f"{share['tracked_calls']} tracked calls; {share['note']})"
                )
            elif share:
                items.append(f"Share of tracked calls scanned: {share['note']}")
            parts.append(_html_list(items))
            parts.append(
                _html_table(
                    ("PII type", "Events"),
                    sorted(dp["pii_events_by_type"].items()),
                    empty="No PII events recorded in the period.",
                )
            )
            parts.append(
                _html_table(
                    ("Redaction strategy", "Records"),
                    sorted(dp["redaction_strategy_distribution"].items()),
                    empty="No redaction strategy recorded.",
                )
            )

        parts.append("<h2>Oversight and audit-trail continuity</h2>")
        ov = self.oversight
        if ov is None:
            parts.append("<p>Audit log not provided; no oversight evidence available.</p>")
        else:
            gaps = ov["gaps_over_threshold"]
            items = [
                f"Records in scope: {ov['records']}",
                f"First record: {ov['first_record'] or 'n/a'}",
                f"Last record: {ov['last_record'] or 'n/a'}",
                f"Records by direction: {_kv_inline(ov['by_direction']) or 'none'}",
                f"Records by method: {_kv_inline(ov['by_method']) or 'none'}",
                f"Gaps over {ov['gap_threshold_hours']}h threshold: {len(gaps)}",
            ]
            if ov["records_without_timestamp"]:
                items.append(
                    f"Records without a parseable timestamp: {ov['records_without_timestamp']}"
                )
            parts.append(_html_list(items))
            if gaps:
                parts.append(
                    _html_table(
                        ("From", "To", "Hours"),
                        [(g["from"], g["to"], g["hours"]) for g in gaps],
                    )
                )
            parts.append(f'<p class="note">{e(ov["note"])}</p>')

        parts.append("<h2>Cost governance</h2>")
        cg = self.cost_governance
        if cg is None:
            parts.append("<p>Costs log not provided; no cost governance evidence available.</p>")
        else:
            items = [
                f"Tracked calls in scope: {cg['calls']}",
                f"Total spend: ${cg['total_cost']:.6f}",
                f"Budget-block events: {cg['budget_block_events']}",
            ]
            if cg["budget_block_note"]:
                items.append(f"Note: {cg['budget_block_note']}")
            if cg["estimated_calls"]:
                items.append(f"Calls with estimated token counts: {cg['estimated_calls']}")
            if cg["unpriced_calls"]:
                items.append(f"Unpriced calls recorded at $0: {cg['unpriced_calls']}")
            parts.append(_html_list(items))
            parts.append(
                _html_table(
                    ("Tag", "Calls", "Cost (USD)"),
                    [
                        (tag, g["calls"], f"{g['cost']:.6f}")
                        for tag, g in cg["spend_by_tag"].items()
                    ],
                    empty="No cost records in the period.",
                )
            )
            parts.append(
                _html_table(
                    ("Model", "Calls", "Cost (USD)"),
                    [
                        (model, g["calls"], f"{g['cost']:.6f}")
                        for model, g in cg["spend_by_model"].items()
                    ],
                    empty="No cost records in the period.",
                )
            )

        parts.append("<h2>AI asset inventory</h2>")
        inv = self.ai_inventory
        if inv is None:
            parts.append(
                "<p>Inventory scan not provided; no AI asset inventory evidence available.</p>"
            )
        else:
            items = [
                f"Scan root: {inv['root'] or 'n/a'}",
                f"Findings: {inv['findings']}",
                f"Providers in use: {', '.join(inv['providers']) or 'none'}",
                "External data-flow providers: "
                + (", ".join(inv["external_data_flow_providers"]) or "none"),
                f"Hardcoded-key findings: {inv['hardcoded_key_findings']}",
            ]
            if inv["scanned_files"] is not None:
                items.append(f"Files scanned: {inv['scanned_files']}")
            if inv["note"]:
                items.append(f"Note: {inv['note']}")
            parts.append(_html_list(items))

        parts.append("<h2>Framework control-theme mapping</h2>")
        if self.framework_mapping:
            parts.append(
                _html_table(
                    ("Evidence", "Source", "Control themes supported"),
                    [
                        (row["evidence"], row["source"], "; ".join(row["supports"]))
                        for row in self.framework_mapping
                    ],
                )
            )
            parts.append(f'<p class="note">{e(_MAPPING_NOTE)}</p>')
        else:
            parts.append("<p>No evidence sources provided; nothing to map.</p>")

        parts.append("<h2>Disclaimer</h2>")
        parts.append(f'<p class="disclaimer">{e(self.disclaimer)}</p>')
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts)


def _kv_inline(mapping: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(mapping.items()))


def _md_table(headers, rows, empty: Optional[str] = None) -> List[str]:
    rows = list(rows)
    if not rows:
        return [empty] if empty else []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def _html_list(items: List[str]) -> str:
    return "<ul>" + "".join(f"<li>{_html.escape(i)}</li>" for i in items) + "</ul>"


def _html_table(headers, rows, empty: Optional[str] = None) -> str:
    rows = list(rows)
    if not rows:
        return f"<p>{_html.escape(empty)}</p>" if empty else ""
    head = "".join(f"<th>{_html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_evidence_report(
    audit_log: Optional[Union[str, Path]] = None,
    costs_log: Optional[Union[str, Path]] = None,
    inventory: Optional[Any] = None,
    period: Optional[str] = None,
    organization: Optional[str] = None,
    gap_threshold_hours: float = 24.0,
    generated_at: Optional[str] = None,
) -> EvidenceReport:
    """Build an :class:`EvidenceReport` from whichever artifacts are provided.

    Each source is optional; absent sources are stated in the report rather
    than inferred. ``period`` filters records by ISO-8601 timestamp prefix
    (e.g. ``"2026-07"``). ``inventory`` accepts an
    :class:`~multimind.observability.ai_inventory.InventoryReport` or its
    ``to_dict()`` output.
    """
    if inventory is not None and hasattr(inventory, "to_dict"):
        inventory = inventory.to_dict()

    audit_records: Optional[List[Dict[str, Any]]] = None
    cost_records: Optional[List[Dict[str, Any]]] = None
    if audit_log is not None:
        records, _ = _read_jsonl(audit_log)
        audit_records = [r for r in records if _in_period(r, period)]
    if costs_log is not None:
        records, _ = _read_jsonl(costs_log)
        cost_records = [r for r in records if _in_period(r, period)]

    sources: Dict[str, Optional[str]] = {
        "audit log": str(audit_log) if audit_log is not None else None,
        "costs log": str(costs_log) if costs_log is not None else None,
        "inventory scan": (
            str((inventory or {}).get("root") or "provided") if inventory is not None else None
        ),
    }
    missing: List[str] = []
    if audit_records is None:
        missing.append(
            "Audit log not provided; data protection and oversight sections have no evidence."
        )
    if cost_records is None:
        missing.append("Costs log not provided; cost governance section has no evidence.")
    if inventory is None:
        missing.append("Inventory scan not provided; AI asset inventory section has no evidence.")

    data_protection = None
    oversight = None
    if audit_records is not None:
        data_protection = _data_protection_section(audit_records)
        data_protection["scanned_call_share"] = _scanned_share(audit_records, cost_records)
        oversight = _oversight_section(audit_records, gap_threshold_hours)

    cost_governance = _cost_governance_section(cost_records) if cost_records is not None else None
    ai_inventory = _inventory_section(inventory) if inventory is not None else None

    mapping = _framework_mapping(
        has_audit=bool(audit_records),
        has_blocked=bool(data_protection and data_protection["blocked_requests"]),
        has_costs=bool(cost_records),
        has_inventory=inventory is not None,
    )

    return EvidenceReport(
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        organization=organization,
        period=period,
        sources=sources,
        missing_sources=missing,
        data_protection=data_protection,
        oversight=oversight,
        cost_governance=cost_governance,
        ai_inventory=ai_inventory,
        framework_mapping=mapping,
    )
