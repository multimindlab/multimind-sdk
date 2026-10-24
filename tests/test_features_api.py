"""Every SDK component reachable through the backend: routes in features_api.

Models are stubbed by monkeypatching ``features_api.get_model_handler``; no
network, no keys.
"""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
from fastapi.testclient import TestClient

import multimind.gateway.api as gw
import multimind.gateway.features_api as fa
from multimind.core.models import ModelResponse


class StubLLM:
    """BaseLLM-shaped stub with deterministic, prompt-aware replies."""

    model_name = "stub-model"
    PROVIDER_NAME = "stub"

    def __init__(self):
        self.prompts = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "question/answer pairs" in prompt:
            return json.dumps(
                [
                    {"question": "What does the guard mask?", "answer": "The guard masks emails."},
                    {"question": "What does the guard mask?", "answer": "duplicate"},
                ]
            )
        if "Extract the factual claims" in prompt:
            return json.dumps(
                [
                    {"claim": "Paris is the capital of France", "verdict": "supported"},
                    {"claim": "Paris has 80 million people", "verdict": "contradicted"},
                ]
            )
        return "stub generation"

    async def chat(self, messages, **kwargs) -> str:
        return "stub summary of the conversation"

    async def embeddings(self, text, **kwargs):
        return [0.0, 1.0]


class StubHandler:
    """Gateway handler that exposes a StubLLM, as SDKModelHandler does."""

    def __init__(self, llm=None):
        self._llm = llm or StubLLM()


class LegacyHandler:
    """Gateway handler without ``_llm`` (OpenAIHandler shape)."""

    config = SimpleNamespace(model_name="legacy-model")

    async def chat(self, messages, **kwargs):
        return ModelResponse(content="legacy chat", model="legacy-model")

    async def generate(self, prompt, **kwargs):
        return ModelResponse(content="legacy generation", model="legacy-model")


@pytest.fixture(autouse=True)
def _fresh_state():
    fa.reset_memory_store()
    fa.reset_privacy_budgets()
    yield
    fa.reset_memory_store()
    fa.reset_privacy_budgets()


@pytest.fixture()
def stub():
    return StubLLM()


@pytest.fixture()
def client(monkeypatch, stub):
    def handler(model):
        if model == "missing":
            raise ValueError("Unsupported model: missing")
        if model == "huggingface":
            raise ImportError("Transformers and PyTorch are required")
        if model == "legacy":
            return LegacyHandler()
        return StubHandler(stub)

    monkeypatch.setattr(fa, "get_model_handler", handler)
    return TestClient(gw.app)


NEW_ROUTES = {
    ("post", "/v1/evaluation/hallucination"),
    ("post", "/v1/sessions/{session_id}/switch"),
    ("post", "/v1/context-transfer"),
    ("post", "/v1/agents/run"),
    ("get", "/v1/memory"),
    ("post", "/v1/memory/{kind}/messages"),
    ("get", "/v1/memory/{kind}/messages"),
    ("delete", "/v1/memory/{kind}"),
    ("get", "/v1/vector-stores"),
    ("post", "/v1/finetune/synthetic-qa"),
    ("get", "/v1/finetune/capabilities"),
    ("get", "/v1/convert/formats"),
    ("post", "/v1/compliance/regulatory-check"),
    ("post", "/v1/compliance/dp/privatize"),
    ("get", "/v1/mcp/tools"),
    ("post", "/v1/mcp/tools/{name}"),
    ("get", "/v1/integrations"),
    ("get", "/v1/traces"),
    ("post", "/v1/traces/demo"),
    ("get", "/v1/features"),
}


# --- registry ---------------------------------------------------------------


def test_features_registry_covers_every_new_route_and_every_path_exists(client):
    body = client.get("/v1/features").json()
    listed = {
        (method.lower(), path)
        for feature in body["features"]
        for method, path in (route.split(" ", 1) for route in feature["routes"])
    }
    assert NEW_ROUTES - {("get", "/v1/features")} <= listed

    gateway_paths = gw.app.openapi()["paths"]
    for method, path in NEW_ROUTES:
        assert path in gateway_paths and method in gateway_paths[path], (method, path)

    from multimind.server.platform import create_platform_app

    platform_paths = create_platform_app().openapi()["paths"]
    for method, path in listed:
        assert path in platform_paths and method in platform_paths[path], (method, path)

    by_name = {f["name"]: f for f in body["features"]}
    assert by_name["hallucination"]["available"] is True
    torch_here = importlib.util.find_spec("torch") is not None
    assert by_name["fine-tuning"]["available"] is torch_here
    if not torch_here:
        assert "torch" in by_name["fine-tuning"]["reason"]
        assert by_name["fine-tuning"]["install"] == fa.FINETUNE_INSTALL


# --- evaluation -------------------------------------------------------------


def test_hallucination_grounding_only(client):
    r = client.post(
        "/v1/evaluation/hallucination",
        json={
            "response": "Paris is the capital of France. It has 80 million people.",
            "sources": ["Paris is the capital and largest city of France."],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["claims"] is None
    assert len(body["sentences"]) == 2
    assert body["sentences"][0]["verdict"] == "supported"
    assert body["sentences"][1]["verdict"] == "unsupported"
    assert body["unsupported_count"] == 1


def test_hallucination_with_judge_model(client):
    r = client.post(
        "/v1/evaluation/hallucination",
        json={
            "response": "Paris is the capital of France.",
            "sources": ["x"],
            "judge_model": "stub",
        },
    )
    assert r.status_code == 200
    claims = r.json()["claims"]
    assert claims["score"] == 0.5
    assert [c["verdict"] for c in claims["claims"]] == ["supported", "contradicted"]


def test_hallucination_judge_bad_output_is_502(client, stub, monkeypatch):
    async def bad(prompt, **kwargs):
        return "not json"

    monkeypatch.setattr(stub, "generate", bad)
    r = client.post(
        "/v1/evaluation/hallucination",
        json={"response": "x.", "sources": ["x"], "judge_model": "stub"},
    )
    assert r.status_code == 502


def test_unknown_model_is_400_and_missing_extra_is_501(client):
    body = {"response": "x.", "sources": ["x"], "judge_model": "missing"}
    assert client.post("/v1/evaluation/hallucination", json=body).status_code == 400
    body["judge_model"] = "huggingface"
    r = client.post("/v1/evaluation/hallucination", json=body)
    assert r.status_code == 501
    assert "huggingface" in r.json()["detail"]


# --- session switch ---------------------------------------------------------


def _session_with_history(client):
    session_id = client.post("/v1/sessions", json={"model": "stub"}).json()["session_id"]
    session = gw.chat_manager.get_session(session_id)
    session.add_message(role="user", content="My name is Ada.", model="stub")
    session.add_message(role="assistant", content="Hello Ada.", model="stub")
    return session_id


def test_session_switch_full_keeps_history(client):
    session_id = _session_with_history(client)
    r = client.post(
        f"/v1/sessions/{session_id}/switch", json={"model": "legacy", "strategy": "full"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "legacy"
    assert body["strategy_applied"] == "full"
    assert body["message_count"] == 2
    session = client.get(f"/v1/sessions/{session_id}").json()
    assert session["model"] == "legacy"
    assert [m["content"] for m in session["messages"]] == ["My name is Ada.", "Hello Ada."]


def test_session_switch_summary_compacts_history_and_records_transfer(client):
    session_id = _session_with_history(client)
    r = client.post(
        f"/v1/sessions/{session_id}/switch", json={"model": "legacy", "strategy": "summary"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["transfer"]["from_model"] == "stub"
    assert body["transfer"]["to_model"] == "legacy"
    assert body["transfer"]["message_count"] == 2
    session = gw.chat_manager.get_session(session_id)
    assert session.model == "legacy"
    assert len(session.messages) == 1
    assert session.messages[0].role == "system"
    assert "stub summary of the conversation" in session.messages[0].content
    assert session.metadata["transfers"][0]["strategy"] == "summary"


def test_session_switch_errors(client):
    assert client.post("/v1/sessions/nope/switch", json={"model": "stub"}).status_code == 404
    session_id = _session_with_history(client)
    r = client.post(f"/v1/sessions/{session_id}/switch", json={"model": "stub", "strategy": "x"})
    assert r.status_code == 400
    r = client.post(f"/v1/sessions/{session_id}/switch", json={"model": "missing"})
    assert r.status_code == 400


# --- context transfer -------------------------------------------------------


def test_context_transfer(client):
    r = client.post(
        "/v1/context-transfer",
        json={
            "messages": [
                {"role": "user", "content": "Plan a trip to Rome."},
                {"role": "assistant", "content": "Day 1: Colosseum."},
            ],
            "source_model": "chatgpt",
            "target_model": "claude",
            "strategy": "structured",
            "last_n": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "Colosseum" in body["formatted_prompt"]
    assert body["metadata"]["summary_type"] == "structured"
    assert body["metadata"]["messages_extracted"] == 2


def test_context_transfer_bad_strategy(client):
    r = client.post(
        "/v1/context-transfer",
        json={"messages": [], "source_model": "a", "target_model": "b", "strategy": "nope"},
    )
    assert r.status_code == 400


# --- agents -----------------------------------------------------------------


class _ToolCallingLLM(StubLLM):
    """Stub with an OpenAI-shaped client: first turn calls the calculator, then answers."""

    def __init__(self):
        super().__init__()
        self.turns = 0
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create))
        )

    async def _create(self, **kwargs):
        self.turns += 1
        if self.turns == 1:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="calculator", arguments=json.dumps({"expression": "17*23"})
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            tool_msgs = [m for m in kwargs["messages"] if m.get("role") == "tool"]
            message = SimpleNamespace(
                content=f"The answer is {tool_msgs[-1]['content']}", tool_calls=None
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_run_native_tool_loop(client, monkeypatch):
    monkeypatch.setattr(fa, "get_model_handler", lambda model: StubHandler(_ToolCallingLLM()))
    r = client.post("/v1/agents/run", json={"model": "stub", "task": "What is 17 * 23?"})
    assert r.status_code == 200
    body = r.json()
    assert body["native_tools"] is True
    assert body["answer"] == "The answer is 391.0"
    assert body["tool_calls"] == [
        {"tool": "calculator", "args": {"expression": "17*23"}, "result": 391.0}
    ]


def test_agent_run_keyword_fallback(client):
    r = client.post("/v1/agents/run", json={"model": "stub", "task": "Say hi", "tools": []})
    assert r.status_code == 200
    assert r.json() == {
        "answer": "stub generation",
        "error": None,
        "native_tools": False,
        "tool_calls": [],
        "tools": [],
    }


def test_agent_run_unknown_tool_is_400(client):
    r = client.post("/v1/agents/run", json={"model": "stub", "task": "x", "tools": ["browser"]})
    assert r.status_code == 400


# --- memory -----------------------------------------------------------------


def test_memory_kinds_construct(client):
    kinds = {k["kind"]: k for k in client.get("/v1/memory").json()["kinds"]}
    assert set(kinds) == {"buffer", "summary", "episodic", "semantic", "procedural"}
    assert all(k["constructible"] for k in kinds.values()), kinds
    assert kinds["buffer"]["needs_model"] is False and kinds["summary"]["needs_model"] is True


def test_buffer_memory_round_trip(client):
    assert client.get("/v1/memory/buffer/messages").json() == {
        "kind": "buffer",
        "active": False,
        "messages": [],
    }
    r = client.post("/v1/memory/buffer/messages", json={"content": "remember this"})
    assert r.status_code == 200 and r.json()["message_count"] == 1
    body = client.get("/v1/memory/buffer/messages").json()
    assert body["active"] is True
    assert body["messages"] == [{"role": "user", "content": "remember this"}]
    assert client.get("/v1/memory").json()["kinds"][0]["active"] is True
    assert client.delete("/v1/memory/buffer").json() == {"kind": "buffer", "cleared": True}
    assert client.get("/v1/memory/buffer/messages").json()["messages"] == []


def test_summary_memory_needs_model_then_uses_it(client, stub):
    assert client.post("/v1/memory/summary/messages", json={"content": "x"}).status_code == 400
    r = client.post(
        "/v1/memory/summary/messages", json={"content": "Ada likes tea.", "model": "stub"}
    )
    assert r.status_code == 200
    assert stub.prompts, "SummaryMemory did not call the model"
    assert client.get("/v1/memory/summary/messages").json()["active"] is True


def test_unknown_memory_kind_is_404(client):
    assert client.get("/v1/memory/quantum/messages").status_code == 404


# --- vector stores ----------------------------------------------------------


def test_vector_store_inventory(client):
    body = client.get("/v1/vector-stores").json()
    assert body["total"] == 60
    by_name = {b["name"]: b for b in body["backends"]}
    faiss = by_name["FAISSBackend"]
    assert faiss["local"] is True and faiss["client_module"] == "faiss"
    assert faiss["constructible"] is faiss["client_installed"]
    assert by_name["PineconeBackend"]["client_module"] == "pinecone"
    assert by_name["PineconeBackend"]["local"] is False
    assert by_name["EpsillaBackend"]["experimental"] is True
    assert by_name["EpsillaBackend"]["constructible"] is False
    for backend in body["backends"]:
        assert isinstance(backend["client_installed"], bool)
        assert backend["constructible"] or backend["reason"]
    assert client.get("/v1/vector-stores").json() == body


# --- fine-tuning and conversion ---------------------------------------------


def test_synthetic_qa(client):
    r = client.post(
        "/v1/finetune/synthetic-qa",
        json={"chunks": ["The guard masks emails."], "model": "stub", "max_pairs": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1  # duplicate question dropped
    assert body["pairs"][0]["question"] == "What does the guard mask?"
    assert body["records"][0]["instruction"] == "What does the guard mask?"


def test_synthetic_qa_bad_model_output_is_502(client, stub, monkeypatch):
    async def bad(prompt, **kwargs):
        return "no json here"

    monkeypatch.setattr(stub, "generate", bad)
    r = client.post("/v1/finetune/synthetic-qa", json={"chunks": ["x"], "model": "stub"})
    assert r.status_code == 502


def test_synthetic_qa_missing_extra_is_501(client):
    r = client.post("/v1/finetune/synthetic-qa", json={"chunks": ["x"], "model": "huggingface"})
    assert r.status_code == 501


def test_finetune_capabilities_report_torch_honestly(client):
    body = client.get("/v1/finetune/capabilities").json()
    torch_here = importlib.util.find_spec("torch") is not None
    assert body["torch_installed"] is torch_here
    assert body["methods"]["synthetic_qa"]["available"] is True
    assert body["methods"]["lora"]["available"] is torch_here
    if not torch_here:
        assert "torch" in body["methods"]["lora"]["reason"]
    assert body["install"] == fa.FINETUNE_INSTALL


def test_convert_formats_is_200_either_way(client):
    r = client.get("/v1/convert/formats")
    assert r.status_code == 200
    body = r.json()
    if body["available"]:
        assert "gguf" in body["targets"]
    else:
        assert body["install"] == fa.FINETUNE_INSTALL
        assert body["reason"]


# --- compliance -------------------------------------------------------------


def test_regulatory_check_offline_detects_change(client, tmp_path):
    state = str(tmp_path / "state.json")
    first = client.post(
        "/v1/compliance/regulatory-check",
        json={"sources": [{"name": "act", "content": "Article 1"}], "state_path": state},
    )
    assert first.status_code == 200
    assert first.json()["events"][0]["changed"] is False
    second = client.post(
        "/v1/compliance/regulatory-check",
        json={"sources": [{"name": "act", "content": "Article 1 amended"}], "state_path": state},
    )
    event = second.json()["events"][0]
    assert event["changed"] is True and "+Article 1 amended" in event["diff"]


def test_regulatory_check_validation(client):
    r = client.post("/v1/compliance/regulatory-check", json={"sources": [{"name": "a"}]})
    assert r.status_code == 400
    r = client.post(
        "/v1/compliance/regulatory-check", json={"sources": [{"name": "a"}], "offline": False}
    )
    assert r.status_code == 400


def test_dp_privatize_values_and_budget(client):
    r = client.post("/v1/compliance/dp/privatize", json={"values": [1, 2], "epsilon": 1.0})
    assert r.status_code == 200
    assert len(r.json()["values"]) == 2 and "budget" not in r.json()

    r = client.post(
        "/v1/compliance/dp/privatize",
        json={"value": 10, "epsilon": 0.6, "budget": 1.0, "budget_id": "t"},
    )
    assert r.status_code == 200
    assert r.json()["budget"]["remaining_epsilon"] == pytest.approx(0.4)
    exhausted = client.post(
        "/v1/compliance/dp/privatize",
        json={"value": 10, "epsilon": 0.6, "budget": 1.0, "budget_id": "t"},
    )
    assert exhausted.status_code == 400
    assert "Privacy budget exhausted" in exhausted.json()["detail"]
    assert client.post("/v1/compliance/dp/privatize", json={"epsilon": 1}).status_code == 400


# --- mcp --------------------------------------------------------------------


def test_mcp_tools_list_and_invoke(client, tmp_path, monkeypatch):
    body = client.get("/v1/mcp/tools").json()
    assert {t["name"] for t in body["tools"]} == {
        "scan_text",
        "redact_text",
        "check_grounding",
        "audit_log",
        "check_policy",
    }
    assert body["tools"][0]["parameters"]["text"]["required"] is True

    scan = client.post(
        "/v1/mcp/tools/scan_text", json={"arguments": {"text": "mail jane@example.com"}}
    )
    assert scan.json()["result"]["total"] == 1
    redact = client.post(
        "/v1/mcp/tools/redact_text", json={"arguments": {"text": "mail jane@example.com"}}
    )
    assert "jane@example.com" not in redact.json()["result"]["redacted"]
    grounding = client.post(
        "/v1/mcp/tools/check_grounding",
        json={"arguments": {"response": "Paris is in France.", "sources": ["Paris is in France."]}},
    )
    assert grounding.json()["result"]["score"] == 1.0
    policy = client.post(
        "/v1/mcp/tools/check_policy", json={"arguments": {"text": "hello", "block_on": ["email"]}}
    )
    assert policy.json()["result"]["allowed"] is True

    monkeypatch.setenv("MULTIMIND_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    audit = client.post("/v1/mcp/tools/audit_log", json={"arguments": {"event": "demo"}})
    assert audit.json()["result"]["ok"] is True and (tmp_path / "audit.jsonl").exists()

    assert client.post("/v1/mcp/tools/nope", json={}).status_code == 404
    assert client.post("/v1/mcp/tools/scan_text", json={"arguments": {}}).status_code == 400


# --- integrations -----------------------------------------------------------


def test_integrations_inventory(client):
    frameworks = {f["framework"]: f for f in client.get("/v1/integrations").json()["frameworks"]}
    assert set(frameworks) == {
        "langchain",
        "llamaindex",
        "crewai",
        "autogen",
        "haystack",
        "openai_client",
    }
    assert "guard_runnable" in frameworks["langchain"]["entry_points"]
    assert frameworks["openai_client"]["entry_points"] == ["guard_openai"]
    for fw in frameworks.values():
        assert isinstance(fw["installed"], bool) and fw["install"].startswith("pip install ")


# --- tracing ----------------------------------------------------------------


def test_traces_without_db_is_honest(client, monkeypatch):
    monkeypatch.delenv("MULTIMIND_TRACING_DB", raising=False)
    body = client.get("/v1/traces").json()
    assert body["available"] is False and body["runs"] == []


def test_trace_demo_records_hashes_only_and_lists_from_sqlite(client, tmp_path, monkeypatch):
    from multimind.observability.tracing import reset_default_tracer

    db = tmp_path / "traces.db"
    monkeypatch.setenv("MULTIMIND_TRACING_DB", str(db))
    reset_default_tracer()
    try:
        r = client.post("/v1/traces/demo", json={"prompt": "secret prompt"})
        assert r.status_code == 200
        body = r.json()
        assert body["persisted"] is True
        assert body["run"]["name"] == "demo-stub.generate"
        assert body["run"]["inputs"] is None and body["run"]["outputs"] is None
        assert body["run"]["input_sha256"]
        assert "secret prompt" not in db.read_bytes().decode("latin-1")

        listed = client.get("/v1/traces").json()
        assert listed["available"] is True
        assert [run["name"] for run in listed["runs"]] == ["demo-stub.generate"]
    finally:
        reset_default_tracer()
