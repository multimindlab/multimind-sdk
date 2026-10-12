"""Tests for the shadow-AI inventory scanner (multimind.observability.ai_inventory)."""

import json

import pytest
from click.testing import CliRunner

from multimind.cli.audit import audit
from multimind.observability.ai_inventory import (
    AI_PACKAGE_REGISTRY,
    InventoryReport,
    scan_project,
)

# Synthetic, non-functional key material (never a real secret).
FAKE_OPENAI_KEY = "sk-" + "abcd1234" * 3
FAKE_ENV_VALUE = "sk-" + "envsecret999" * 2


@pytest.fixture
def project(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "openai>=1.0\nlangchain-core==0.2.0\nrequests\n", encoding="utf-8"
    )
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"@anthropic-ai/sdk": "^0.20.0", "left-pad": "1.0.0"}}),
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "import os\n"
        f'API_KEY = "{FAKE_OPENAI_KEY}"\n'
        'key = os.environ["OPENAI_API_KEY"]\n'
        'BASE = "https://api.openai.com/v1"\n',
        encoding="utf-8",
    )
    hidden = tmp_path / "node_modules" / "somelib"
    hidden.mkdir(parents=True)
    (hidden / "index.js").write_text(
        'fetch("https://api.anthropic.com/v1/messages")\n', encoding="utf-8"
    )
    return tmp_path


# --- registry / dependency matching ---------------------------------------------


def test_registry_entries_have_required_fields():
    for name, info in AI_PACKAGE_REGISTRY.items():
        assert info["kind"] in ("sdk", "framework"), name
        assert info["data_flow"] in ("external", "local"), name
        assert info["provider"], name


def test_dependency_matching(project):
    report = scan_project(project)
    deps = {f.evidence: f for f in report.findings if f.kind == "dependency"}
    assert "openai" in deps
    assert deps["openai"].file == "requirements.txt"
    assert deps["openai"].line == 1
    assert "langchain-core" in deps  # prefix family match
    assert deps["langchain-core"].provider == "multi"
    assert "@anthropic-ai/sdk" in deps
    assert "requests" not in deps
    assert "left-pad" not in deps


# --- source scan -----------------------------------------------------------------


def test_source_findings(project):
    report = scan_project(project)
    keys = [f for f in report.findings if f.kind == "hardcoded_key"]
    assert len(keys) == 1
    assert keys[0].file == "app.py"
    assert keys[0].line == 2
    assert keys[0].provider == "openai"

    env_reads = [f for f in report.findings if f.kind == "env_read"]
    assert [(f.evidence, f.file, f.line) for f in env_reads] == [("OPENAI_API_KEY", "app.py", 3)]

    urls = [f for f in report.findings if f.kind == "api_url"]
    assert [(f.provider, f.file, f.line) for f in urls] == [("openai", "app.py", 4)]


def test_node_modules_skipped(project):
    report = scan_project(project)
    assert all("node_modules" not in (f.file or "") for f in report.findings)
    assert not any(f.kind == "api_url" and f.provider == "anthropic" for f in report.findings)


def test_no_raw_key_values_anywhere(project):
    report = scan_project(project)
    dumped = json.dumps(report.to_dict())
    assert FAKE_OPENAI_KEY not in dumped
    assert "abcd1234" not in dumped
    for f in report.findings:
        assert FAKE_OPENAI_KEY not in f.evidence


# --- report ------------------------------------------------------------------


def test_summary_and_risks(project):
    report = scan_project(project)
    assert isinstance(report, InventoryReport)
    summary = report.summary()
    assert summary["by_kind"]["dependency"] == 3
    assert summary["by_kind"]["hardcoded_key"] == 1
    assert summary["by_kind"]["env_read"] == 1
    assert summary["by_kind"]["api_url"] == 1
    assert summary["by_provider"]["openai"] == 4
    assert summary["by_data_flow"]["external"] == len(report.findings)
    assert summary["scanned_files"] == 3  # requirements.txt, package.json, app.py

    risks = report.risks()
    assert "openai" in risks["external_data_flow_providers"]
    assert "anthropic" in risks["external_data_flow_providers"]
    assert len(risks["hardcoded_keys"]) == 1


def test_skipped_files_reported_honestly(tmp_path):
    (tmp_path / "big.py").write_text("x = 1\n" * 500, encoding="utf-8")
    (tmp_path / "small.py").write_text("import os\n", encoding="utf-8")
    report = scan_project(tmp_path, max_file_bytes=100)
    assert report.skipped_files == 1
    assert report.scanned_files == 1
    assert "skipped 1 file(s)" in report.to_dict()["note"]


def test_max_files_cap(tmp_path):
    for i in range(3):
        (tmp_path / f"mod{i}.py").write_text("import os\n", encoding="utf-8")
    report = scan_project(tmp_path, max_files=1)
    assert report.scanned_files == 1
    assert report.skipped_files == 2


def test_scan_project_rejects_non_directory(tmp_path):
    target = tmp_path / "file.py"
    target.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        scan_project(target)


def test_include_env_names_only(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_ENV_VALUE)
    report = scan_project(tmp_path, include_env=True)
    env_set = [f for f in report.findings if f.kind == "env_set"]
    assert any(f.evidence == "OPENAI_API_KEY" and f.provider == "openai" for f in env_set)
    assert FAKE_ENV_VALUE not in json.dumps(report.to_dict())


# --- CLI -----------------------------------------------------------------------


def test_cli_scan_exit_1_on_hardcoded_key(project):
    runner = CliRunner()
    result = runner.invoke(audit, ["scan", str(project)])
    assert result.exit_code == 1
    assert "Hardcoded API key" in result.output
    assert FAKE_OPENAI_KEY not in result.output


def test_cli_scan_json_output(project):
    runner = CliRunner()
    result = runner.invoke(audit, ["scan", str(project), "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["summary"]["by_kind"]["hardcoded_key"] == 1
    assert FAKE_OPENAI_KEY not in result.output


def test_cli_scan_exit_0_without_keys(tmp_path):
    (tmp_path / "requirements.txt").write_text("openai\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(audit, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_cli_scan_empty_project(tmp_path):
    runner = CliRunner()
    result = runner.invoke(audit, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "No AI usage found" in result.output


def test_cli_costs_from_jsonl(tmp_path):
    from multimind.observability.cost_tracker import CostTracker

    path = tmp_path / "costs.jsonl"
    tracker = CostTracker(jsonl_path=path)
    tracker.record("openai", "gpt-4o", 100, 50, 0.003, tag="team-a")
    tracker.record("claude", "claude-3", 10, 5, 0.001)
    runner = CliRunner()
    result = runner.invoke(audit, ["costs", "--log", str(path)])
    assert result.exit_code == 0
    assert "Chargeback report" in result.output
    assert "team-a" in result.output
    assert "(untagged)" in result.output
    assert "TOTAL" in result.output
