"""Tests for run tracing (multimind.observability.tracing)."""

import hashlib
import json
import sqlite3

import httpx
import pytest

from multimind.compliance.guard import guard
from multimind.observability.cost_tracker import CostTracker, track_costs
from multimind.observability.tracing import (
    RunTracer,
    TracedModel,
    get_default_tracer,
    reset_default_tracer,
    trace_model,
)

CONTRACT_FIELDS = {
    "id",
    "project",
    "name",
    "run_type",
    "start_time",
    "end_time",
    "status",
    "error",
    "model",
    "provider",
    "input_chars",
    "output_chars",
    "input_sha256",
    "output_sha256",
    "inputs",
    "outputs",
    "tokens",
    "cost",
    "tags",
    "parent_id",
}


class MockModel:
    """Minimal BaseLLM-compatible async model with flat pricing."""

    PROVIDER_NAME = "MockProvider"

    def __init__(self, response="four word mock reply", cost_per_token=0.00001, chunks=None):
        self.model_name = "mock-model"
        self.cost_per_token = cost_per_token
        self.response = response
        self.chunks = chunks or ["chunk one ", "chunk two"]
        self.calls = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        return self.response

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return self.response

    async def generate_stream(self, prompt, **kwargs):
        self.calls += 1
        for chunk in self.chunks:
            yield chunk

    def generate_sync(self, prompt, **kwargs):
        self.calls += 1
        return self.response


class UsageModel(MockModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}


class FakePost:
    """Capturing httpx.post replacement; optionally fails the first N calls."""

    def __init__(self, fail_first=0, status_code=200):
        self.calls = []
        self.fail_first = fail_first
        self.status_code = status_code

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.fail_first > 0:
            self.fail_first -= 1
            raise httpx.ConnectError("connection refused")
        return httpx.Response(self.status_code, request=httpx.Request("POST", url))


@pytest.fixture
def fake_post(monkeypatch):
    fake = FakePost()
    monkeypatch.setattr("multimind.observability.tracing.httpx.post", fake)
    return fake


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_run_lifecycle():
    tracer = RunTracer(project="proj")
    run = tracer.start_run("job", "chain", inputs="hello")
    run.end(outputs="world")
    assert len(tracer.completed) == 1
    d = tracer.completed[0]["run"]
    assert d["status"] == "success"
    assert d["error"] is None
    assert d["project"] == "proj"
    assert d["run_type"] == "chain"
    assert d["end_time"] >= d["start_time"]
    assert d["parent_id"] is None


def test_span_nesting():
    tracer = RunTracer()
    root = tracer.start_run("root", "chain")
    child = root.span("child", "tool")
    grandchild = child.span("grandchild", "llm")
    grandchild.end()
    child.end()
    root.end()
    payload = tracer.completed[0]
    spans = {s["name"]: s for s in payload["spans"]}
    assert spans["child"]["parent_id"] == payload["run"]["id"]
    assert spans["grandchild"]["parent_id"] == spans["child"]["id"]
    assert spans["child"]["run_type"] == "tool"


def test_invalid_run_type():
    tracer = RunTracer()
    with pytest.raises(ValueError):
        tracer.start_run("bad", "embedding")


def test_privacy_default_no_content(tmp_path, fake_post):
    db = tmp_path / "runs.db"
    secret = "ssn is 123-45-6789"
    reply = "redacted reply"
    tracer = RunTracer(export_url="https://platform.example", api_key="k", sqlite_path=db)
    run = tracer.start_run("job", "llm", inputs=secret)
    run.end(outputs=reply)
    d = tracer.completed[0]["run"]
    assert d["inputs"] is None and d["outputs"] is None
    assert d["input_chars"] == len(secret) and d["output_chars"] == len(reply)
    assert d["input_sha256"] == sha(secret) and d["output_sha256"] == sha(reply)
    assert secret not in json.dumps(fake_post.calls[0]["json"])
    row = sqlite3.connect(db).execute("SELECT inputs, outputs FROM runs").fetchone()
    assert row == (None, None)
    blob = db.read_bytes()
    assert secret.encode() not in blob and reply.encode() not in blob


def test_store_content_opt_in(tmp_path):
    db = tmp_path / "runs.db"
    tracer = RunTracer(sqlite_path=db, store_content=True)
    run = tracer.start_run("job", "llm", inputs={"prompt": "hello"})
    run.end(outputs="world")
    d = tracer.completed[0]["run"]
    assert d["inputs"] == {"prompt": "hello"}
    assert d["outputs"] == "world"
    row = sqlite3.connect(db).execute("SELECT inputs, outputs FROM runs").fetchone()
    assert json.loads(row[0]) == {"prompt": "hello"}
    assert json.loads(row[1]) == "world"


def test_sqlite_rows_match_contract(tmp_path):
    db = tmp_path / "runs.db"
    tracer = RunTracer(project="p", sqlite_path=db, tags=["team-a"])
    root = tracer.start_run("root", "chain", inputs="in")
    span = root.span("step", "tool")
    span.end()
    root.end(outputs="out")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = {r["name"]: dict(r) for r in conn.execute("SELECT * FROM runs")}
    assert set(rows["root"]) == CONTRACT_FIELDS
    assert rows["root"]["parent_id"] is None
    assert rows["step"]["parent_id"] == rows["root"]["id"]
    assert json.loads(rows["root"]["tags"]) == ["team-a"]
    assert rows["root"]["input_chars"] == 2
    assert rows["root"]["input_sha256"] == sha("in")
    assert rows["root"]["status"] == "success"


def test_export_payload_contract(fake_post):
    tracer = RunTracer(project="proj", export_url="https://platform.example/", api_key="secret-key")
    root = tracer.start_run("root", "chain", inputs="q")
    span = root.span("llm-step", "llm", model="m", provider="p")
    span.end(outputs="a", tokens={"input": 3, "output": 5, "estimated": True}, cost=0.01)
    root.end(outputs="a")
    assert len(fake_post.calls) == 1
    call = fake_post.calls[0]
    assert call["url"] == "https://platform.example/api/v1/runs"
    assert call["headers"]["X-API-Key"] == "secret-key"
    body = call["json"]
    assert set(body) == {"run", "spans"}
    assert set(body["run"]) == CONTRACT_FIELDS
    assert body["run"]["parent_id"] is None
    assert len(body["spans"]) == 1
    s = body["spans"][0]
    assert set(s) == CONTRACT_FIELDS
    assert s["parent_id"] == body["run"]["id"]
    assert s["tokens"] == {"input": 3, "output": 5, "estimated": True}
    assert s["cost"] == 0.01
    assert s["model"] == "m" and s["provider"] == "p"


def test_export_failure_buffers_and_reflushes(monkeypatch):
    fake = FakePost(fail_first=1)
    monkeypatch.setattr("multimind.observability.tracing.httpx.post", fake)
    tracer = RunTracer(export_url="https://platform.example", api_key="k")
    tracer.start_run("job").end()
    assert tracer.pending_exports == 1
    sent = tracer.flush()
    assert sent == 1
    assert tracer.pending_exports == 0
    assert len(fake.calls) == 2
    assert fake.calls[0]["json"] == fake.calls[1]["json"]


def test_export_http_error_buffers(monkeypatch):
    fake = FakePost(status_code=500)
    monkeypatch.setattr("multimind.observability.tracing.httpx.post", fake)
    tracer = RunTracer(export_url="https://platform.example")
    tracer.start_run("job").end()
    assert tracer.pending_exports == 1
    fake.status_code = 200
    assert tracer.flush() == 1
    assert tracer.pending_exports == 0


async def test_traced_model_records_llm_run():
    tracer = RunTracer()
    model = trace_model(MockModel(), tracer=tracer)
    result = await model.generate("test prompt")
    assert result == "four word mock reply"
    d = tracer.completed[0]["run"]
    assert d["run_type"] == "llm"
    assert d["name"] == "mock-model.generate"
    assert d["model"] == "mock-model" and d["provider"] == "MockProvider"
    assert d["end_time"] >= d["start_time"]
    assert d["tokens"] == {"input": 3, "output": 5, "estimated": True}
    assert d["cost"] == pytest.approx(8 * 0.00001)
    assert d["inputs"] is None and d["input_chars"] == len("test prompt")


async def test_traced_model_real_usage_and_stream():
    tracer = RunTracer()
    model = trace_model(UsageModel(), tracer=tracer)
    await model.chat([{"role": "user", "content": "hi"}])
    d = tracer.completed[0]["run"]
    assert d["tokens"] == {"input": 100, "output": 50, "estimated": False}
    chunks = [c async for c in model.generate_stream("go")]
    assert chunks == ["chunk one ", "chunk two"]
    d = tracer.completed[1]["run"]
    assert d["output_chars"] == len("chunk one chunk two")


async def test_traced_model_error_run():
    class FailingModel(MockModel):
        async def generate(self, prompt, **kwargs):
            raise RuntimeError("boom")

    tracer = RunTracer()
    model = trace_model(FailingModel(), tracer=tracer)
    with pytest.raises(RuntimeError):
        await model.generate("x")
    d = tracer.completed[0]["run"]
    assert d["status"] == "error"
    assert "boom" in d["error"]


def test_traced_model_sync_and_delegation():
    tracer = RunTracer()
    model = trace_model(MockModel(), tracer=tracer, name="my-llm")
    assert model.generate_sync("p") == "four word mock reply"
    assert model.model_name == "mock-model"
    d = tracer.completed[0]["run"]
    assert d["name"] == "my-llm.generate_sync"


async def test_composes_with_guard_and_track_costs():
    cost_tracker = CostTracker()
    tracer = RunTracer()
    model = trace_model(track_costs(guard(MockModel()), tracker=cost_tracker), tracer=tracer)
    assert isinstance(model, TracedModel)
    result = await model.generate("hello there")
    assert result == "four word mock reply"
    d = tracer.completed[0]["run"]
    assert d["model"] == "mock-model" and d["provider"] == "MockProvider"
    assert d["status"] == "success"
    assert len(cost_tracker.records) == 1
    assert cost_tracker.records[0].model == "mock-model"


async def test_decorator_sync_and_async():
    tracer = RunTracer()

    @tracer.traced("sync-step", run_type="tool")
    def add(a, b):
        return a + b

    @tracer.traced
    async def fetch(x):
        return f"got {x}"

    assert add(1, 2) == 3
    assert await fetch("y") == "got y"
    names = [(p["run"]["name"], p["run"]["run_type"]) for p in tracer.completed]
    assert names == [("sync-step", "tool"), ("fetch", "chain")]


def test_context_manager_nesting_and_error():
    tracer = RunTracer()
    with tracer.run("pipeline", inputs="q") as root:
        with root.span("step") as step:
            step.end(outputs="ok")
    payload = tracer.completed[0]
    assert payload["run"]["status"] == "success"
    assert payload["spans"][0]["parent_id"] == payload["run"]["id"]
    with pytest.raises(ValueError):
        with tracer.run("failing"):
            raise ValueError("bad input")
    d = tracer.completed[1]["run"]
    assert d["status"] == "error"
    assert "bad input" in d["error"]


def test_ambient_nesting_of_traced_model():
    tracer = RunTracer()
    model = trace_model(MockModel(), tracer=tracer)
    with tracer.run("chain") as root:
        model.generate_sync("p")
        root.end()
    payload = tracer.completed[0]
    assert payload["spans"][0]["name"] == "mock-model.generate_sync"
    assert payload["spans"][0]["parent_id"] == payload["run"]["id"]


def test_default_tracer_env(monkeypatch, tmp_path):
    db = tmp_path / "default.db"
    monkeypatch.setenv("MULTIMIND_TRACING_PROJECT", "env-proj")
    monkeypatch.setenv("MULTIMIND_PLATFORM_URL", "https://platform.example")
    monkeypatch.setenv("MULTIMIND_PLATFORM_KEY", "env-key")
    monkeypatch.setenv("MULTIMIND_TRACING_DB", str(db))
    reset_default_tracer()
    try:
        tracer = get_default_tracer()
        assert tracer is get_default_tracer()
        assert tracer.project == "env-proj"
        assert tracer.export_url == "https://platform.example"
        assert tracer.api_key == "env-key"
        assert tracer.sqlite_path == db
        model = trace_model(MockModel())
        assert model.tracer is tracer
    finally:
        reset_default_tracer()
