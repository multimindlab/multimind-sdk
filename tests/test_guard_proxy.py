"""Tests for the OpenAI-compatible compliance proxy (multimind.gateway.guard_proxy).

Upstream is mocked with httpx.MockTransport; no live API calls.
"""

import json

import pytest

pytest.importorskip("fastapi", reason="requires multimind-sdk[gateway]")
import httpx
from fastapi.testclient import TestClient

import multimind.gateway.guard_proxy as gp

EMAIL = "bob@example.com"
SSN = "212-45-6789"

CHAT_BODY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

EMBEDDING_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

MODELS_BODY = {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _stream_chunk(content=None, finish_reason=None):
    delta = {"content": content} if content is not None else {}
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _default_stream_body():
    # An email split across chunk boundaries plus a finish chunk.
    parts = ["Sure, contact bob@", "example.com for", " details."]
    body = _sse(_stream_chunk(content=""))
    for part in parts:
        body += _sse(_stream_chunk(content=part))
    body += _sse(_stream_chunk(finish_reason="stop"))
    body += "data: [DONE]\n\n"
    return body


@pytest.fixture()
def upstream(monkeypatch):
    """Mock upstream that records every request it receives."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            if payload.get("stream"):
                return httpx.Response(
                    200,
                    content=_default_stream_body().encode(),
                    headers={"content-type": "text/event-stream"},
                )
            body = json.loads(json.dumps(CHAT_BODY))
            body["model"] = payload.get("model", "gpt-4o")
            return httpx.Response(200, json=body)
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=EMBEDDING_BODY)
        if path.endswith("/models"):
            return httpx.Response(200, json=MODELS_BODY)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        gp,
        "_async_client",
        lambda state: httpx.AsyncClient(base_url=state.base_url, transport=transport),
    )
    return seen


def make_client(**settings_kwargs):
    settings_kwargs.setdefault("upstream_base_url", "http://upstream.test/v1")
    settings = gp.ProxySettings(**settings_kwargs)
    app = gp.create_app(settings)
    return TestClient(app), app


def chat_payload(content, **extra):
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": content}], **extra}


def test_health_and_ready(upstream):
    client, _ = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["upstream"] == "http://upstream.test/v1"


def test_list_models_forwarded(upstream):
    client, _ = make_client()
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == MODELS_BODY
    assert upstream[-1].url.path == "/v1/models"


def test_chat_round_trip_preserves_openai_shape(upstream):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload("Hello"))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "chatcmpl-123"
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello there"
    assert body["usage"]["total_tokens"] == 30


def test_pii_redacted_before_upstream(upstream):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload(f"My email is {EMAIL}."))
    assert r.status_code == 200
    sent = json.loads(upstream[-1].content)
    content = sent["messages"][0]["content"]
    assert EMAIL not in content
    assert "[EMAIL]" in content


def test_pii_redacted_in_multimodal_parts(upstream):
    client, _ = make_client()
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Reach me at {EMAIL}"}],
            }
        ],
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    sent = json.loads(upstream[-1].content)
    text = sent["messages"][0]["content"][0]["text"]
    assert EMAIL not in text
    assert "[EMAIL]" in text


def test_block_on_returns_400(upstream):
    client, _ = make_client(block_on="ssn,credit_card")
    r = client.post("/v1/chat/completions", json=chat_payload(f"My SSN is {SSN}"))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "compliance_violation"
    assert err["code"] == "pii_blocked"
    assert "ssn" in err["blocked_types"]
    assert SSN not in r.text
    # Nothing was forwarded upstream.
    assert upstream == []


def test_budget_exhaustion_returns_429(upstream):
    # gpt-4o blended price makes one 30-token call exceed this ceiling.
    client, app = make_client(budget=0.0001)
    r1 = client.post("/v1/chat/completions", json=chat_payload("Hello"))
    assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json=chat_payload("Hello again"))
    assert r2.status_code == 429
    err = r2.json()["error"]
    assert err["type"] == "budget_exceeded"
    assert err["spent"] > 0
    # Only the first request reached the upstream.
    assert len(upstream) == 1
    assert app.state.proxy.tracker.total_cost > 0


def test_audit_jsonl_written_without_raw_pii(upstream, tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    client, _ = make_client(audit_log=str(audit_path))
    r = client.post("/v1/chat/completions", json=chat_payload(f"Contact {EMAIL} please"))
    assert r.status_code == 200
    raw = audit_path.read_text()
    assert EMAIL not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    directions = {rec["direction"] for rec in records}
    assert {"input", "output"} <= directions
    inp = next(rec for rec in records if rec["direction"] == "input")
    assert inp["pii_types"] == {"email": 1}
    assert inp["count"] == 1
    assert inp["blocked"] is False
    assert inp["tags"] and all(tag.startswith("[EMAIL:") for tag in inp["tags"])


def test_blocked_request_audited(upstream, tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    client, _ = make_client(block_on="ssn", audit_log=str(audit_path))
    r = client.post("/v1/chat/completions", json=chat_payload(f"SSN: {SSN}"))
    assert r.status_code == 400
    raw = audit_path.read_text()
    assert SSN not in raw
    rec = json.loads(raw.splitlines()[0])
    assert rec["blocked"] is True
    assert rec["pii_types"] == {"ssn": 1}


def test_response_scanning_redacts_output(upstream, monkeypatch):
    body = json.loads(json.dumps(CHAT_BODY))
    body["choices"][0]["message"]["content"] = f"Write to {EMAIL} for help"

    def handler(request):
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        gp,
        "_async_client",
        lambda state: httpx.AsyncClient(base_url=state.base_url, transport=transport),
    )
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload("Who do I contact?"))
    content = r.json()["choices"][0]["message"]["content"]
    assert EMAIL not in content
    assert "[EMAIL]" in content


def test_scan_output_off_passes_response_through(upstream, monkeypatch):
    body = json.loads(json.dumps(CHAT_BODY))
    body["choices"][0]["message"]["content"] = f"Write to {EMAIL}"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    monkeypatch.setattr(
        gp,
        "_async_client",
        lambda state: httpx.AsyncClient(base_url=state.base_url, transport=transport),
    )
    client, _ = make_client(scan_output=False)
    r = client.post("/v1/chat/completions", json=chat_payload("Who?"))
    assert r.json()["choices"][0]["message"]["content"] == f"Write to {EMAIL}"


def test_streaming_yields_valid_sse_and_scans_output(upstream):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload("Who?", stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in r.text.split("\n") if line.startswith("data:")]
    assert lines[-1].strip() == "data: [DONE]"
    content = ""
    finish_reasons = []
    for line in lines[:-1]:
        obj = json.loads(line[len("data:") :])
        assert obj["object"] == "chat.completion.chunk"
        choice = obj["choices"][0]
        content += choice["delta"].get("content") or ""
        finish_reasons.append(choice.get("finish_reason"))
    # Email split across upstream chunks is still caught and redacted.
    assert "bob@example.com" not in content
    assert "[EMAIL]" in content
    assert "for details." in content
    assert "stop" in finish_reasons


def test_streaming_pii_in_request_redacted(upstream):
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload(f"I am {EMAIL}", stream=True))
    assert r.status_code == 200
    sent = json.loads(upstream[-1].content)
    assert EMAIL not in sent["messages"][0]["content"]


def test_streaming_records_estimated_cost(upstream):
    client, app = make_client()
    client.post("/v1/chat/completions", json=chat_payload("Who?", stream=True))
    records = app.state.proxy.tracker.records
    assert len(records) == 1
    assert records[0].method == "chat.completions.stream"
    assert records[0].estimated is True
    assert records[0].output_tokens > 0


def test_embeddings_redacted_and_forwarded(upstream):
    client, _ = make_client()
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": f"Contact {EMAIL} now"},
    )
    assert r.status_code == 200
    assert r.json() == EMBEDDING_BODY
    sent = json.loads(upstream[-1].content)
    assert EMAIL not in sent["input"]
    assert "[EMAIL]" in sent["input"]


def test_embeddings_list_input_and_block(upstream):
    client, _ = make_client(block_on="email")
    r = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["hello", f"mail {EMAIL}"]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "pii_blocked"
    assert upstream == []


def test_upstream_error_passthrough(upstream, monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    monkeypatch.setattr(
        gp,
        "_async_client",
        lambda state: httpx.AsyncClient(base_url=state.base_url, transport=transport),
    )
    client, _ = make_client()
    r = client.post("/v1/chat/completions", json=chat_payload("Hello"))
    assert r.status_code == 401
    assert r.json()["error"]["message"] == "bad key"


def test_upstream_auth_header_injected(upstream, monkeypatch):
    client, _ = make_client(upstream_api_key="sk-upstream-test-key-000000")
    client.post("/v1/chat/completions", json=chat_payload("Hello"))
    auth = upstream[-1].headers.get("authorization")
    assert auth == "Bearer sk-upstream-test-key-000000"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MULTIMIND_UPSTREAM_BASE_URL", "http://env-upstream/v1")
    monkeypatch.setenv("MULTIMIND_PROXY_STRATEGY", "hash")
    monkeypatch.setenv("MULTIMIND_PROXY_BLOCK_ON", "ssn, credit_card")
    monkeypatch.setenv("MULTIMIND_PROXY_BUDGET", "5.0")
    monkeypatch.setenv("MULTIMIND_PROXY_SCAN_OUTPUT", "false")
    monkeypatch.setenv("MULTIMIND_PROXY_PORT", "9999")
    settings = gp.ProxySettings.from_env(strategy="remove")
    assert settings.upstream_base_url == "http://env-upstream/v1"
    assert settings.strategy == "remove"  # explicit override wins
    assert settings.block_on == ("ssn", "credit_card")
    assert settings.budget == 5.0
    assert settings.scan_output is False
    assert settings.port == 9999


def test_resolve_upstream_named_providers(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    base_url, api_key = gp.resolve_upstream(gp.ProxySettings(upstream="groq"))
    assert base_url == "https://api.groq.com/openai/v1"
    assert api_key is None
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    base_url, api_key = gp.resolve_upstream(gp.ProxySettings(upstream="ollama"))
    assert base_url == "http://localhost:11434/v1"
    with pytest.raises(ValueError, match="Unknown upstream"):
        gp.resolve_upstream(gp.ProxySettings(upstream="nope"))


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError):
        gp.ProxySettings(strategy="shred")


def test_serve_cli_help():
    from click.testing import CliRunner

    from multimind.cli.serve import serve

    result = CliRunner().invoke(serve, ["--help"])
    assert result.exit_code == 0
    assert "compliance proxy" in result.output
