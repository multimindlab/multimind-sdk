"""CliRunner tests for every MultiMind CLI command group. No live API calls."""

import importlib
import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from multimind.cli import cli
from multimind.cli.context_transfer import main as context_transfer_main

# The package __init__ rebinds these attribute names to the click Groups,
# so grab the actual modules for monkeypatching.
chat_module = importlib.import_module("multimind.cli.chat")
models_module = importlib.import_module("multimind.cli.models")
compliance_module = importlib.import_module("multimind.cli.compliance")


def _extras_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# Gateway/compliance-backed commands exit 1 on core-only installs by design;
# skip their tests when the extras are not installed (they run on the CI jobs
# that install [gateway] / [compliance]).
requires_gateway = pytest.mark.skipif(
    not _extras_available("multimind.gateway.models"),
    reason="requires the [gateway] extras",
)
requires_compliance_backend = pytest.mark.skipif(
    not _extras_available("multimind.gateway.compliance_api"),
    reason="requires the [compliance,gateway] extras",
)

ALL_KEY_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "DEEPSEEK_API_KEY",
]


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def no_keys(monkeypatch):
    for var in ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class FakeResponse:
    def __init__(self, content="mocked response", usage=None):
        self.content = content
        self.usage = usage or {"total_tokens": 3}


class FakeHandler:
    async def generate(self, prompt, **kwargs):
        return FakeResponse()

    async def chat(self, messages, **kwargs):
        return FakeResponse()


def _fake_monitor():
    # The real gateway monitor works fine against FakeHandler in tests
    from multimind.gateway.monitoring import monitor

    return monitor


def _fake_backend(**overrides):
    from multimind.compliance.governance import GovernanceConfig, Regulation

    backend = {
        "GovernanceConfig": GovernanceConfig,
        "Regulation": Regulation,
        "generate_compliance_report": None,
        "get_compliance_alerts": None,
        "get_dashboard_metrics": None,
        "run_compliance_monitoring": None,
        "save_alert_rules": None,
    }
    backend.update(overrides)
    return backend


# ---------------------------------------------------------------- help tree


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["chat", "--help"],
        ["models", "--help"],
        ["config", "--help"],
        ["compliance", "--help"],
    ],
)
def test_help_screens(runner, args):
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_unknown_command(runner):
    result = runner.invoke(cli, ["definitely-not-a-command"])
    assert result.exit_code == 2


# ---------------------------------------------------------------- chat


def test_chat_start_missing_key_fails_fast(runner, no_keys):
    result = runner.invoke(cli, ["chat", "start", "-m", "openai", "-p", "hi"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY not set" in result.output


def test_chat_start_single_prompt_mocked(runner, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(chat_module, "_gateway", lambda: (None, lambda m: FakeHandler()))
    result = runner.invoke(cli, ["chat", "start", "-m", "openai", "-p", "hi"])
    assert result.exit_code == 0
    assert "mocked response" in result.output


@requires_gateway
def test_chat_start_unsupported_model(runner):
    result = runner.invoke(cli, ["chat", "start", "-m", "notamodel", "-p", "hi"])
    assert result.exit_code == 1
    assert "Unsupported model" in result.output


def test_chat_start_requires_model_option(runner):
    result = runner.invoke(cli, ["chat", "start"])
    assert result.exit_code == 2


def test_chat_list_sessions_empty(runner, monkeypatch):
    mgr = SimpleNamespace(list_sessions=lambda: [])
    monkeypatch.setattr(chat_module, "_gateway", lambda: (mgr, None))
    result = runner.invoke(cli, ["chat", "list-sessions"])
    assert result.exit_code == 0
    assert "No active sessions" in result.output


def test_chat_load_missing_session(runner, monkeypatch):
    mgr = SimpleNamespace(get_session=lambda s: None, load_session=lambda s: None)
    monkeypatch.setattr(chat_module, "_gateway", lambda: (mgr, None))
    result = runner.invoke(cli, ["chat", "load", "nosuch"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_chat_save_missing_session(runner, monkeypatch):
    mgr = SimpleNamespace(save_session=lambda s: False)
    monkeypatch.setattr(chat_module, "_gateway", lambda: (mgr, None))
    result = runner.invoke(cli, ["chat", "save", "nosuch"])
    assert result.exit_code == 1


def test_chat_delete_missing_session(runner, monkeypatch):
    mgr = SimpleNamespace(delete_session=lambda s: False)
    monkeypatch.setattr(chat_module, "_gateway", lambda: (mgr, None))
    result = runner.invoke(cli, ["chat", "delete", "nosuch"])
    assert result.exit_code == 1


def test_chat_delete_existing_session(runner, monkeypatch):
    mgr = SimpleNamespace(delete_session=lambda s: True)
    monkeypatch.setattr(chat_module, "_gateway", lambda: (mgr, None))
    result = runner.invoke(cli, ["chat", "delete", "sess1"])
    assert result.exit_code == 0
    assert "Deleted session" in result.output


# ---------------------------------------------------------------- models


def test_models_list_offline_shows_all_providers(runner, no_keys):
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    for provider in ["openai", "claude", "ollama", "groq", "mistral", "gemini", "deepseek"]:
        assert provider in result.output
    assert "not set" in result.output


def test_models_list_shows_configured_keys(runner, no_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "configured" in result.output


def test_models_list_local_dir(runner, tmp_path):
    (tmp_path / "my-model").mkdir()
    result = runner.invoke(cli, ["models", "list", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "my-model" in result.output


def test_models_list_local_dir_missing(runner, tmp_path):
    result = runner.invoke(cli, ["models", "list", "--output-dir", str(tmp_path / "nope")])
    assert result.exit_code == 0
    assert "No models found" in result.output


@requires_gateway
def test_models_compare_no_keys_fails(runner, no_keys):
    result = runner.invoke(cli, ["models", "compare", "hi", "-m", "openai"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY not set" in result.output
    assert "No model produced a response" in result.output


@requires_gateway
def test_models_compare_mocked(runner, no_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        models_module, "_gateway", lambda: (lambda m: FakeHandler(), _fake_monitor())
    )
    result = runner.invoke(cli, ["models", "compare", "hi", "-m", "openai"])
    assert result.exit_code == 0
    assert "mocked response" in result.output


@requires_gateway
def test_models_metrics(runner):
    result = runner.invoke(cli, ["models", "metrics"])
    assert result.exit_code == 0
    assert "Model Metrics" in result.output


@requires_gateway
def test_models_metrics_single_model(runner):
    result = runner.invoke(cli, ["models", "metrics", "-m", "openai"])
    assert result.exit_code == 0
    assert "openai" in result.output


def test_models_health_missing_key(runner, no_keys):
    result = runner.invoke(cli, ["models", "health", "-m", "openai"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY not set" in result.output


@requires_gateway
def test_models_health_mocked(runner, no_keys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        models_module, "_gateway", lambda: (lambda m: FakeHandler(), _fake_monitor())
    )
    result = runner.invoke(cli, ["models", "health", "-m", "openai"])
    assert result.exit_code == 0
    assert "openai Health Check" in result.output


def test_models_health_no_configured_models(runner, no_keys):
    result = runner.invoke(cli, ["models", "health"])
    assert result.exit_code == 0
    assert "No models with configured API keys" in result.output


def test_models_delete_aborted(runner, tmp_path):
    target = tmp_path / "victim"
    target.mkdir()
    result = runner.invoke(cli, ["models", "delete", "-m", str(target)], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert target.exists()


def test_models_delete_confirmed(runner, tmp_path):
    target = tmp_path / "victim"
    target.mkdir()
    result = runner.invoke(cli, ["models", "delete", "-m", str(target)], input="y\n")
    assert result.exit_code == 0
    assert not target.exists()


def test_models_export_missing_path(runner, tmp_path):
    result = runner.invoke(cli, ["models", "export", "-m", str(tmp_path / "nope")])
    assert result.exit_code == 2


# ---------------------------------------------------------------- config


def test_config_info(runner, home):
    result = runner.invoke(cli, ["config", "info"])
    assert result.exit_code == 0
    assert "Python version" in result.output


def test_config_manage_set_get(runner, home):
    result = runner.invoke(cli, ["config", "manage", "--set", "theme", "dark"])
    assert result.exit_code == 0
    result = runner.invoke(cli, ["config", "manage", "--get", "theme"])
    assert result.exit_code == 0
    assert "dark" in result.output


def test_config_manage_show_all(runner, home):
    result = runner.invoke(cli, ["config", "manage"])
    assert result.exit_code == 0
    assert "Key" in result.output


def test_config_manage_corrupt_file(runner, home):
    (home / ".multimind_cli_config").write_text("{not json")
    result = runner.invoke(cli, ["config", "manage"])
    assert result.exit_code == 1
    assert "JSON" in result.output


def test_config_completion_zsh(runner):
    result = runner.invoke(cli, ["config", "completion", "zsh"])
    assert result.exit_code == 0
    assert "multimind" in result.output


def test_config_completion_bad_shell(runner):
    result = runner.invoke(cli, ["config", "completion", "powershell"])
    assert result.exit_code == 2


# ---------------------------------------------------------------- compliance


def test_compliance_run_compliance_requires_config(runner):
    result = runner.invoke(cli, ["compliance", "run-compliance"])
    assert result.exit_code == 2
    assert "--config" in result.output


def test_compliance_generate_report_requires_config(runner):
    result = runner.invoke(cli, ["compliance", "generate-report"])
    assert result.exit_code == 2


def test_compliance_configure_alerts_requires_config(runner):
    result = runner.invoke(cli, ["compliance", "configure-alerts", "-o", "org1"])
    assert result.exit_code == 2


@requires_compliance_backend
def test_compliance_dashboard_offline(runner):
    result = runner.invoke(cli, ["compliance", "dashboard", "-o", "org1"])
    assert result.exit_code == 0
    assert "Compliance Dashboard" in result.output


@requires_compliance_backend
def test_compliance_alerts_offline(runner):
    result = runner.invoke(cli, ["compliance", "alerts", "-o", "org1"])
    assert result.exit_code == 0
    assert "Compliance Alerts" in result.output


def test_compliance_run_compliance_mocked(runner, tmp_path, monkeypatch):
    config = {
        "organization_id": "org1",
        "organization_name": "Org",
        "dpo_email": "dpo@example.com",
        "enabled_regulations": ["GDPR"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    async def fake_monitoring(cfg):
        return {"final_evaluation": {"score": 1.0, "recommendations": []}}

    monkeypatch.setattr(
        compliance_module,
        "_compliance_backend",
        lambda: _fake_backend(run_compliance_monitoring=fake_monitoring),
    )
    result = runner.invoke(cli, ["compliance", "run-compliance", "-c", str(config_path)])
    assert result.exit_code == 0
    assert "Compliance Evaluation Results" in result.output


def test_compliance_configure_alerts_mocked(runner, tmp_path, monkeypatch):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rule": "high"}))

    async def fake_save(org_id, rules):
        return None

    monkeypatch.setattr(
        compliance_module, "_compliance_backend", lambda: _fake_backend(save_alert_rules=fake_save)
    )
    result = runner.invoke(
        cli, ["compliance", "configure-alerts", "-o", "org1", "-c", str(rules_path)]
    )
    assert result.exit_code == 0
    assert "configured successfully" in result.output


# ---------------------------------------------------------------- scan-text


def test_scan_text_finds_pii(runner):
    result = runner.invoke(
        cli,
        ["compliance", "scan-text", "Email john.doe@example.com, SSN 123-45-6789"],
    )
    assert result.exit_code == 1
    assert "PII Findings" in result.output
    assert "email" in result.output
    assert "ssn" in result.output


def test_scan_text_clean(runner):
    result = runner.invoke(cli, ["compliance", "scan-text", "nothing sensitive here"])
    assert result.exit_code == 0
    assert "No PII found" in result.output


def test_scan_text_from_file(runner, tmp_path):
    pii_file = tmp_path / "sample.txt"
    pii_file.write_text("Contact jane@example.com today")
    result = runner.invoke(cli, ["compliance", "scan-text", "--file", str(pii_file)])
    assert result.exit_code == 1
    assert "email" in result.output


def test_scan_text_missing_file(runner, tmp_path):
    result = runner.invoke(cli, ["compliance", "scan-text", "--file", str(tmp_path / "nope.txt")])
    assert result.exit_code == 2


def test_scan_text_no_input(runner):
    result = runner.invoke(cli, ["compliance", "scan-text"])
    assert result.exit_code == 2
    assert "Provide TEXT or --file" in result.output


def test_scan_text_both_inputs(runner, tmp_path):
    pii_file = tmp_path / "sample.txt"
    pii_file.write_text("x")
    result = runner.invoke(cli, ["compliance", "scan-text", "some text", "--file", str(pii_file)])
    assert result.exit_code == 2


def test_scan_text_redact(runner):
    result = runner.invoke(
        cli, ["compliance", "scan-text", "Email john.doe@example.com", "--redact", "mask"]
    )
    assert result.exit_code == 1
    assert "[EMAIL]" in result.output


# ---------------------------------------------------------------- context-transfer


def test_context_transfer_list_models():
    assert context_transfer_main(["--list_models"]) == 0


def test_context_transfer_model_info():
    assert context_transfer_main(["--model_info", "deepseek"]) == 0


def test_context_transfer_missing_args_exits_2():
    with pytest.raises(SystemExit) as exc:
        context_transfer_main(["--from_model", "chatgpt"])
    assert exc.value.code == 2


def test_context_transfer_missing_input_file(tmp_path):
    rc = context_transfer_main(
        [
            "--from_model",
            "chatgpt",
            "--to_model",
            "deepseek",
            "--input_file",
            str(tmp_path / "nope.json"),
            "--output_file",
            str(tmp_path / "out.txt"),
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------- lazy convert import


def test_cli_import_does_not_load_model_conversion():
    # The CLI package must import without the torch-only model_conversion
    # modules (torch itself may still load via guarded optional imports).
    import subprocess
    import sys

    code = (
        "import sys; import multimind.cli; "
        "bad = [m for m in sys.modules if m.startswith('multimind.model_conversion')]; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_convert_without_torch_fails_cleanly(monkeypatch, capsys):
    import builtins

    from multimind.cli import _run_convert

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("multimind.model_conversion"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = _run_convert()
    assert rc == 1
    captured = capsys.readouterr()
    assert "multimind-sdk[finetune]" in captured.err + captured.out
