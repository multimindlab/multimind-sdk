"""Tests for the Anthropic MCP server (multimind.mcp_server)."""

import importlib
import json
import re
import sys

import pytest

from multimind.mcp_server import (
    SERVER_NAME,
    audit_log,
    check_grounding,
    check_policy,
    create_server,
    redact_text,
    scan_text,
)

EMAIL = "jane.doe@example.com"
SSN = "123-45-6789"

TOOL_NAMES = {"scan_text", "redact_text", "check_grounding", "audit_log", "check_policy"}


@pytest.fixture(autouse=True)
def _real_mcp_package():
    """Force the installed MCP SDK to win over any same-named example package
    (historically ``examples/mcp``, renamed to ``examples/mcp_workflows``).
    Kept as defense in depth in case a shadowing ``mcp`` dir reappears."""
    import importlib.util
    import site

    def _real_mcp_importable() -> bool:
        try:
            import mcp.shared.memory  # noqa: F401

            return True
        except Exception:
            return False

    if not _real_mcp_importable():
        for name in [n for n in list(sys.modules) if n == "mcp" or n.startswith("mcp.")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        # Prepend real site-packages so the installed `mcp` wins over any shadow
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)
        if not _real_mcp_importable():
            pytest.skip("installed MCP SDK is shadowed and could not be restored")
    yield


def _data(result):
    """Unwrap a CallToolResult into the tool's dict payload."""
    if result.structuredContent and set(result.structuredContent) == {"result"}:
        return result.structuredContent["result"]
    return json.loads(result.content[0].text)


class TestScanText:
    def test_clean_text_yields_no_findings(self):
        result = scan_text("hello world, nothing sensitive here")
        assert result == {"findings": [], "counts": {}, "total": 0}

    def test_findings_have_spans_but_no_raw_values(self):
        text = f"contact {EMAIL} or ssn {SSN}"
        result = scan_text(text)
        assert result["counts"] == {"email": 1, "ssn": 1}
        assert result["total"] == 2
        for finding in result["findings"]:
            assert set(finding) == {"type", "start", "end"}
        email_span = next(f for f in result["findings"] if f["type"] == "email")
        assert text[email_span["start"] : email_span["end"]] == EMAIL
        assert EMAIL not in json.dumps(result)
        assert SSN not in json.dumps(result)


class TestRedactText:
    def test_mask(self):
        result = redact_text(f"mail {EMAIL} now")
        assert result["redacted"] == "mail [EMAIL] now"
        assert result["counts"] == {"email": 1}
        assert result["total"] == 1

    def test_hash(self):
        result = redact_text(f"ssn {SSN}", strategy="hash")
        assert re.fullmatch(r"ssn \[SSN:[0-9a-f]{8}\]", result["redacted"])

    def test_remove(self):
        result = redact_text(f"mail {EMAIL} now", strategy="remove")
        assert result["redacted"] == "mail  now"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            redact_text("anything", strategy="rot13")


class TestCheckGrounding:
    def test_supported_and_unsupported_sentences(self):
        report = check_grounding(
            "The sky is blue. Elephants can fly.",
            sources=["The sky is blue. Grass is green."],
        )
        verdicts = {s["sentence"]: s["verdict"] for s in report["sentences"]}
        assert verdicts["The sky is blue."] == "supported"
        assert verdicts["Elephants can fly."] == "unsupported"
        assert report["unsupported_count"] == 1
        assert 0.0 < report["score"] < 1.0
        assert "1 supported" in report["summary"]

    def test_empty_response_scores_perfect(self):
        report = check_grounding("", sources=["anything"])
        assert report["score"] == 1.0
        assert report["sentences"] == []
        assert report["unsupported_count"] == 0

    def test_empty_sources_leave_everything_unsupported(self):
        report = check_grounding("The sky is blue.", sources=[])
        assert report["score"] == 0.0
        assert report["unsupported_count"] == 1


class TestAuditLog:
    def test_writes_jsonl_line_without_raw_pii(self, tmp_path, monkeypatch):
        path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("MULTIMIND_AUDIT_LOG", str(path))
        result = audit_log(
            f"user pasted {EMAIL}",
            metadata={"note": f"ssn seen {SSN}", "attempts": 2},
        )
        assert result == {"ok": True, "path": str(path)}
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert EMAIL not in lines[0]
        assert SSN not in lines[0]
        entry = json.loads(lines[0])
        assert entry["source"] == "mcp"
        assert entry["event"] == "user pasted [EMAIL]"
        assert entry["metadata"] == {"note": "ssn seen [SSN]", "attempts": 2}
        assert entry["pii_types"] == {"email": 1, "ssn": 1}
        assert "timestamp" in entry

    def test_appends_and_defaults_metadata(self, tmp_path, monkeypatch):
        path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("MULTIMIND_AUDIT_LOG", str(path))
        audit_log("first event")
        audit_log("second event")
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert "metadata" not in json.loads(lines[0])


class TestCheckPolicy:
    def test_blocked(self):
        result = check_policy(f"ssn {SSN} and mail {EMAIL}", block_on=["ssn"])
        assert result == {"allowed": False, "violations": [{"type": "ssn", "count": 1}]}

    def test_allowed_when_no_blocked_types_present(self):
        result = check_policy(f"mail {EMAIL}", block_on=["ssn", "credit_card"])
        assert result == {"allowed": True, "violations": []}

    def test_block_on_is_case_insensitive(self):
        result = check_policy(f"ssn {SSN}", block_on=["SSN"])
        assert result["allowed"] is False


class TestMCPServer:
    """End-to-end over the real MCP SDK in-memory transport."""

    @pytest.fixture
    def client(self):
        pytest.importorskip("mcp")
        from mcp.shared.memory import create_connected_server_and_client_session

        return create_connected_server_and_client_session(create_server())

    async def test_lists_all_tools(self, client):
        async with client as session:
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == TOOL_NAMES
            assert all(t.description for t in tools.tools)

    async def test_scan_text_roundtrip(self, client):
        async with client as session:
            result = await session.call_tool("scan_text", {"text": f"mail {EMAIL}"})
            assert not result.isError
            data = _data(result)
            assert data["counts"] == {"email": 1}
            assert EMAIL not in json.dumps(data)

    async def test_check_policy_roundtrip(self, client):
        async with client as session:
            result = await session.call_tool(
                "check_policy", {"text": f"ssn {SSN}", "block_on": ["ssn"]}
            )
            data = _data(result)
            assert data["allowed"] is False

    async def test_bad_strategy_is_tool_error(self, client):
        async with client as session:
            result = await session.call_tool("redact_text", {"text": "x", "strategy": "rot13"})
            assert result.isError

    async def test_audit_log_roundtrip(self, client, tmp_path, monkeypatch):
        path = tmp_path / "audit.jsonl"
        monkeypatch.setenv("MULTIMIND_AUDIT_LOG", str(path))
        async with client as session:
            result = await session.call_tool("audit_log", {"event": f"saw {EMAIL}"})
            data = _data(result)
            assert data["ok"] is True
        assert EMAIL not in path.read_text()

    def test_server_name(self):
        assert create_server().name == SERVER_NAME == "multimind-compliance"


class TestImportSafety:
    def test_importable_and_helpful_error_without_mcp(self, monkeypatch):
        for name in [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "mcp", None)  # makes `import mcp` fail
        for name in [n for n in sys.modules if n.startswith("multimind.mcp_server")]:
            monkeypatch.delitem(sys.modules, name)
        module = importlib.import_module("multimind.mcp_server")
        assert module.scan_text("hello")["total"] == 0
        with pytest.raises(ImportError, match=r"multimind-sdk\[mcp\]"):
            module.create_server()
