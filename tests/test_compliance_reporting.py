"""Tests for compliance evidence reporting (multimind.compliance.reporting)."""

import json
from html.parser import HTMLParser

import pytest
from click.testing import CliRunner

from multimind.cli import cli
from multimind.compliance import EvidenceReport, build_evidence_report
from multimind.compliance.reporting import DISCLAIMER

AUDIT_RECORDS = [
    {
        "timestamp": "2026-06-15T09:00:00+00:00",
        "direction": "input",
        "method": "generate",
        "pii_types": {"email": 1},
        "count": 1,
        "strategy": "mask",
        "blocked": False,
        "tags": [],
    },
    {
        "timestamp": "2026-07-01T10:00:00+00:00",
        "direction": "input",
        "method": "generate",
        "pii_types": {"email": 2},
        "count": 2,
        "strategy": "mask",
        "blocked": False,
        "tags": [],
    },
    {
        "timestamp": "2026-07-01T10:05:00+00:00",
        "direction": "output",
        "method": "generate",
        "pii_types": {},
        "count": 0,
        "strategy": "mask",
        "blocked": False,
        "tags": [],
    },
    {
        "timestamp": "2026-07-03T09:00:00+00:00",
        "direction": "input",
        "method": "chat",
        "pii_types": {"ssn": 1},
        "count": 1,
        "strategy": "hash",
        "blocked": True,
        "tags": [],
    },
]

COST_RECORDS = [
    {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.0015,
        "tag": "team-a",
        "method": "generate",
        "estimated": False,
        "unpriced": False,
        "timestamp": "2026-07-01T10:00:01+00:00",
    },
    {
        "provider": "anthropic",
        "model": "claude-3",
        "input_tokens": 200,
        "output_tokens": 80,
        "cost": 0.002,
        "tag": None,
        "method": "chat",
        "estimated": True,
        "unpriced": False,
        "timestamp": "2026-07-03T09:00:01+00:00",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.0001,
        "tag": "team-a",
        "method": "generate",
        "estimated": False,
        "unpriced": False,
        "timestamp": "2026-06-20T12:00:00+00:00",
    },
]

INVENTORY = {
    "root": "/proj",
    "findings": [],
    "summary": {
        "findings": 3,
        "by_provider": {"anthropic": 1, "openai": 2},
        "by_kind": {"dependency": 2, "hardcoded_key": 1},
        "by_data_flow": {"external": 3},
        "scanned_files": 12,
        "skipped_files": 0,
    },
    "risks": {
        "external_data_flow_providers": ["anthropic", "openai"],
        "hardcoded_keys": [{"kind": "hardcoded_key", "provider": "openai"}],
    },
}


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


@pytest.fixture
def audit_log(tmp_path):
    return _write_jsonl(tmp_path / "audit.jsonl", AUDIT_RECORDS)


@pytest.fixture
def costs_log(tmp_path):
    return _write_jsonl(tmp_path / "costs.jsonl", COST_RECORDS)


@pytest.fixture
def runner():
    return CliRunner()


# ------------------------------------------------------------ known answers


def test_data_protection_numbers(audit_log, costs_log):
    report = build_evidence_report(audit_log=audit_log, costs_log=costs_log, period="2026-07")
    dp = report.data_protection
    assert dp["records"] == 3
    assert dp["records_with_pii"] == 2
    assert dp["pii_events_by_type"] == {"email": 2, "ssn": 1}
    assert dp["redaction_strategy_distribution"] == {"hash": 1, "mask": 2}
    assert dp["blocked_requests"] == 1
    assert dp["blocked_pii_types"] == ["ssn"]
    share = dp["scanned_call_share"]
    assert share["scanned_input_records"] == 2
    assert share["tracked_calls"] == 2
    assert share["share_pct"] == 100.0


def test_oversight_numbers(audit_log):
    report = build_evidence_report(audit_log=audit_log, period="2026-07")
    ov = report.oversight
    assert ov["records"] == 3
    assert ov["first_record"] == "2026-07-01T10:00:00+00:00"
    assert ov["last_record"] == "2026-07-03T09:00:00+00:00"
    assert ov["by_direction"] == {"input": 2, "output": 1}
    assert ov["by_method"] == {"chat": 1, "generate": 2}


def test_cost_governance_numbers(costs_log):
    report = build_evidence_report(costs_log=costs_log, period="2026-07")
    cg = report.cost_governance
    assert cg["calls"] == 2
    assert cg["total_cost"] == pytest.approx(0.0035)
    assert cg["spend_by_tag"]["team-a"] == {"calls": 1, "cost": pytest.approx(0.0015)}
    assert cg["spend_by_tag"]["(untagged)"]["calls"] == 1
    assert cg["spend_by_model"]["gpt-4o"]["calls"] == 1
    assert cg["estimated_calls"] == 1
    assert cg["budget_block_events"] == 0
    assert "no budget-block events" in cg["budget_block_note"]


def test_period_filter_excludes_other_months(audit_log, costs_log):
    unfiltered = build_evidence_report(audit_log=audit_log, costs_log=costs_log)
    assert unfiltered.data_protection["records"] == 4
    assert unfiltered.cost_governance["calls"] == 3
    filtered = build_evidence_report(audit_log=audit_log, costs_log=costs_log, period="2026-06")
    assert filtered.data_protection["records"] == 1
    assert filtered.cost_governance["calls"] == 1


def test_inventory_section():
    report = build_evidence_report(inventory=INVENTORY)
    inv = report.ai_inventory
    assert inv["providers"] == ["anthropic", "openai"]
    assert inv["external_data_flow_providers"] == ["anthropic", "openai"]
    assert inv["hardcoded_key_findings"] == 1
    assert inv["findings"] == 3
    assert inv["scanned_files"] == 12


# ------------------------------------------------------------- gap detection


def test_gap_detection(audit_log):
    report = build_evidence_report(audit_log=audit_log, period="2026-07")
    gaps = report.oversight["gaps_over_threshold"]
    assert len(gaps) == 1
    assert gaps[0]["from"] == "2026-07-01T10:05:00+00:00"
    assert gaps[0]["to"] == "2026-07-03T09:00:00+00:00"
    assert gaps[0]["hours"] == pytest.approx(46.9, abs=0.1)


def test_gap_threshold_configurable(audit_log):
    report = build_evidence_report(
        audit_log=audit_log, period="2026-07", gap_threshold_hours=100.0
    )
    assert report.oversight["gaps_over_threshold"] == []


# ------------------------------------------------------ absent-source honesty


def test_absent_costs_log_is_stated_not_fabricated(audit_log):
    report = build_evidence_report(audit_log=audit_log, period="2026-07")
    assert report.cost_governance is None
    md = report.to_markdown()
    assert "costs log not provided" in md.lower()
    assert "total spend" not in md.lower()
    share = report.data_protection["scanned_call_share"]
    assert share["share_pct"] is None
    assert "costs log not provided" in share["note"]


def test_absent_audit_log_is_stated(costs_log):
    report = build_evidence_report(costs_log=costs_log)
    assert report.data_protection is None
    assert report.oversight is None
    md = report.to_markdown()
    assert "audit log not provided" in md.lower()


def test_absent_inventory_is_stated(audit_log):
    report = build_evidence_report(audit_log=audit_log)
    assert report.ai_inventory is None
    assert "inventory scan not provided" in report.to_markdown().lower()


# ---------------------------------------------------------- framework mapping


def test_framework_mapping_reflects_present_sources(audit_log, costs_log):
    report = build_evidence_report(
        audit_log=audit_log, costs_log=costs_log, inventory=INVENTORY, period="2026-07"
    )
    sources = [row["source"] for row in report.framework_mapping]
    assert sources == ["audit log", "audit log", "costs log", "inventory scan"]
    themes = {t for row in report.framework_mapping for t in row["supports"]}
    assert "EU AI Act Art. 12 record-keeping" in themes
    assert "EU AI Act Art. 50 transparency" in themes
    assert "HIPAA 164.312(b) audit controls" in themes
    assert "SOC 2 monitoring" in themes
    assert not any("satisfies" in t for t in themes)


def test_framework_mapping_shrinks_without_sources(costs_log):
    report = build_evidence_report(costs_log=costs_log, period="2026-07")
    sources = [row["source"] for row in report.framework_mapping]
    assert sources == ["costs log"]


# -------------------------------------------------------- disclaimer honesty


def test_disclaimer_always_present(audit_log, costs_log):
    for report in (
        build_evidence_report(audit_log=audit_log),
        build_evidence_report(costs_log=costs_log),
        build_evidence_report(inventory=INVENTORY),
        build_evidence_report(audit_log=audit_log, costs_log=costs_log, inventory=INVENTORY),
    ):
        assert "not a legal compliance determination" in report.disclaimer
        assert DISCLAIMER in report.to_markdown()
        assert DISCLAIMER in report.to_dict()["disclaimer"]
        assert "not a legal compliance determination" in report.to_html()


# ----------------------------------------------------------------- renderers


class _TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        for _, value in attrs:
            assert value is None or "http" not in value


def test_markdown_renders_all_sections(audit_log, costs_log):
    md = build_evidence_report(
        audit_log=audit_log, costs_log=costs_log, inventory=INVENTORY, period="2026-07"
    ).to_markdown()
    for heading in (
        "# AI Compliance Evidence Report",
        "## Sources",
        "## Data protection (PII controls)",
        "## Oversight and audit-trail continuity",
        "## Cost governance",
        "## AI asset inventory",
        "## Framework control-theme mapping",
        "## Disclaimer",
    ):
        assert heading in md


def test_html_renders_and_is_self_contained(audit_log, costs_log):
    html_out = build_evidence_report(
        audit_log=audit_log, costs_log=costs_log, inventory=INVENTORY, period="2026-07"
    ).to_html()
    assert "http://" not in html_out and "https://" not in html_out
    parser = _TagCollector()
    parser.feed(html_out)
    assert "title" in parser.tags
    assert "table" in parser.tags
    assert "style" in parser.tags


def test_renderers_deterministic(audit_log, costs_log):
    kwargs = dict(
        audit_log=audit_log,
        costs_log=costs_log,
        inventory=INVENTORY,
        period="2026-07",
        organization="Acme",
        generated_at="2026-07-07T00:00:00+00:00",
    )
    a = build_evidence_report(**kwargs)
    b = build_evidence_report(**kwargs)
    assert a.to_dict() == b.to_dict()
    assert a.to_markdown() == b.to_markdown()
    assert a.to_html() == b.to_html()


def test_report_is_dataclass_instance(audit_log):
    assert isinstance(build_evidence_report(audit_log=audit_log), EvidenceReport)


# ------------------------------------------------------------------------ CLI


def test_cli_report_evidence_writes_html(runner, tmp_path, audit_log, costs_log):
    out = tmp_path / "evidence.html"
    result = runner.invoke(
        cli,
        [
            "compliance",
            "report-evidence",
            "--audit-log",
            audit_log,
            "--costs-log",
            costs_log,
            "--period",
            "2026-07",
            "--format",
            "html",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "not a legal compliance determination" in text
    assert "AI Compliance Evidence Report" in text


def test_cli_report_evidence_markdown_to_stdout(runner, audit_log):
    result = runner.invoke(cli, ["compliance", "report-evidence", "--audit-log", audit_log])
    assert result.exit_code == 0
    assert "# AI Compliance Evidence Report" in result.output
    assert "costs log not provided" in result.output.lower()


def test_cli_report_evidence_with_project_scan(runner, tmp_path, audit_log):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "requirements.txt").write_text("openai\n", encoding="utf-8")
    result = runner.invoke(
        cli,
        ["compliance", "report-evidence", "--audit-log", audit_log, "--project", str(proj)],
    )
    assert result.exit_code == 0
    assert "openai" in result.output


def test_cli_report_evidence_requires_audit_log(runner):
    result = runner.invoke(cli, ["compliance", "report-evidence"])
    assert result.exit_code != 0
