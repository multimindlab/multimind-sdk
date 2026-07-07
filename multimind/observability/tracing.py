"""Run tracing for MultiMind: local SQLite sink + platform export.

``RunTracer`` records runs (root traces) and nested spans for LLM, chain,
and tool calls, then writes them to a local SQLite database and/or exports
them to the MultiMind platform (``POST {export_url}/api/v1/runs`` with an
``X-API-Key`` header). ``trace_model`` wraps any BaseLLM-compatible model
(same ``__getattr__`` delegation pattern as ``ComplianceGuard`` /
``TrackedModel``) and records one llm run per call with latency, token
usage, and cost.

Privacy-first by default: prompt/response content is neither stored nor
exported — only character counts and SHA-256 hashes, enough to detect
duplicates and payload drift without retaining the text itself. Pass
``store_content=True`` to opt in to keeping inputs/outputs verbatim; that
makes traces far easier to debug but means prompts and completions (and any
PII they contain) land in the SQLite file and the platform payloads.
"""

from __future__ import annotations

import atexit
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .cost_tracker import _extract_usage, _messages_text, _resolve_cost, estimate_tokens

logger = logging.getLogger(__name__)

RUN_TYPES = ("llm", "chain", "tool")

_current_run: contextvars.ContextVar[Optional["Run"]] = contextvars.ContextVar(
    "multimind_current_run", default=None
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_text(obj: Any) -> str:
    """Canonical text form of run content, used for char counts and hashes."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


class Run:
    """One traced unit of work; spans are ``Run`` objects with a parent."""

    def __init__(
        self,
        tracer: "RunTracer",
        name: str,
        run_type: str = "chain",
        inputs: Any = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent: Optional["Run"] = None,
    ):
        if run_type not in RUN_TYPES:
            raise ValueError(f"Unknown run_type: {run_type!r} (use one of {RUN_TYPES})")
        self.id = str(uuid.uuid4())
        self.tracer = tracer
        self.name = name
        self.run_type = run_type
        self.project = tracer.project
        self.start_time = _utcnow()
        self.end_time: Optional[str] = None
        self.status = "success"
        self.error: Optional[str] = None
        self.model = model
        self.provider = provider
        self.tags = list(dict.fromkeys([*tracer.tags, *(tags or [])]))
        self.parent = parent
        self.parent_id = parent.id if parent is not None else None
        self.children: List[Run] = []
        input_text = _content_text(inputs)
        self.input_chars = len(input_text)
        self.input_sha256 = _sha256(input_text)
        self.inputs = _jsonable(inputs) if tracer.store_content and inputs is not None else None
        self.output_chars = 0
        self.output_sha256 = _sha256("")
        self.outputs: Any = None
        self.tokens: Optional[Dict[str, Any]] = None
        self.cost: Optional[float] = None
        self._ended = False
        self._ctx_token: Optional[contextvars.Token] = None
        if parent is not None:
            parent.children.append(self)

    def span(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Any = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "Run":
        """Start a child span of this run."""
        return Run(
            self.tracer,
            name,
            run_type,
            inputs=inputs,
            model=model,
            provider=provider,
            tags=tags,
            parent=self,
        )

    def end(
        self,
        outputs: Any = None,
        error: Optional[str] = None,
        tokens: Optional[Dict[str, Any]] = None,
        cost: Optional[float] = None,
    ) -> None:
        """Finish the run; a finished root run is dispatched to the sinks."""
        if self._ended:
            return
        self._ended = True
        self.end_time = _utcnow()
        if error is not None:
            self.status = "error"
            self.error = str(error)
        output_text = _content_text(outputs)
        self.output_chars = len(output_text)
        self.output_sha256 = _sha256(output_text)
        if self.tracer.store_content and outputs is not None:
            self.outputs = _jsonable(outputs)
        if tokens is not None:
            self.tokens = {
                "input": int(tokens.get("input", 0)),
                "output": int(tokens.get("output", 0)),
                "estimated": bool(tokens.get("estimated", False)),
            }
        if cost is not None:
            self.cost = float(cost)
        if self.parent is None:
            self.tracer._finalize(self)

    def __enter__(self) -> "Run":
        self._ctx_token = _current_run.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ctx_token is not None:
            _current_run.reset(self._ctx_token)
            self._ctx_token = None
        if exc is not None:
            self.end(error=f"{exc_type.__name__}: {exc}")
        else:
            self.end()

    def to_dict(self) -> Dict[str, Any]:
        """Contract-shaped dict as sent to ``POST /api/v1/runs``."""
        return {
            "id": self.id,
            "project": self.project,
            "name": self.name,
            "run_type": self.run_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "error": self.error,
            "model": self.model,
            "provider": self.provider,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tokens": self.tokens,
            "cost": self.cost,
            "tags": self.tags,
            "parent_id": self.parent_id,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    name TEXT NOT NULL,
    run_type TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    status TEXT NOT NULL,
    error TEXT,
    model TEXT,
    provider TEXT,
    input_chars INTEGER NOT NULL,
    output_chars INTEGER NOT NULL,
    input_sha256 TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    inputs TEXT,
    outputs TEXT,
    tokens TEXT,
    cost REAL,
    tags TEXT NOT NULL,
    parent_id TEXT REFERENCES runs(id)
)
"""

_INSERT = """
INSERT OR REPLACE INTO runs (
    id, project, name, run_type, start_time, end_time, status, error,
    model, provider, input_chars, output_chars, input_sha256, output_sha256,
    inputs, outputs, tokens, cost, tags, parent_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class RunTracer:
    """Records runs/spans and dispatches completed traces to the sinks.

    Sinks: ``sqlite_path`` writes runs and spans to a local SQLite file;
    ``export_url`` POSTs each completed root run (with its spans) to
    ``{export_url}/api/v1/runs`` with the ``X-API-Key`` header. Export
    failures are logged and the payload stays buffered for the next
    :meth:`flush` (also registered via ``atexit``) — tracing never crashes
    the host app.

    Privacy: by default only char counts and SHA-256 hashes of inputs and
    outputs are recorded. ``store_content=True`` opts in to storing and
    exporting prompt/response content verbatim — useful for debugging, but
    it means raw text (including any PII) is retained locally and sent to
    the platform.
    """

    def __init__(
        self,
        project: str = "default",
        export_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sqlite_path: Optional[Union[str, Path]] = None,
        store_content: bool = False,
        tags: Optional[List[str]] = None,
    ):
        self.project = project
        self.export_url = export_url.rstrip("/") if export_url else None
        self.api_key = api_key
        self.sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
        self.store_content = store_content
        self.tags = list(tags or [])
        self.completed: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.sqlite_path is not None:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(_SCHEMA)
        if self.export_url is not None:
            atexit.register(self.flush)

    def start_run(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Any = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent: Optional[Run] = None,
    ) -> Run:
        """Start a run; nests under the ambient run when one is active."""
        if parent is None:
            ambient = _current_run.get()
            if ambient is not None and ambient.tracer is self and not ambient._ended:
                parent = ambient
        return Run(
            self,
            name,
            run_type,
            inputs=inputs,
            model=model,
            provider=provider,
            tags=tags,
            parent=parent,
        )

    def run(self, name: str, run_type: str = "chain", **kwargs) -> Run:
        """Context-manager form: ``with tracer.run("step") as r: ...``."""
        return self.start_run(name, run_type, **kwargs)

    def traced(self, name: Any = None, run_type: str = "chain", tags: Optional[List[str]] = None):
        """Decorator recording one run per call (sync or async functions)."""

        def decorator(fn):
            run_name = name if isinstance(name, str) else fn.__name__

            def _inputs(args, kwargs):
                return {"args": list(args), "kwargs": kwargs}

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def awrapper(*args, **kwargs):
                    with self.run(run_name, run_type, inputs=_inputs(args, kwargs), tags=tags) as r:
                        result = await fn(*args, **kwargs)
                        r.end(outputs=result)
                        return result

                return awrapper

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with self.run(run_name, run_type, inputs=_inputs(args, kwargs), tags=tags) as r:
                    result = fn(*args, **kwargs)
                    r.end(outputs=result)
                    return result

            return wrapper

        if callable(name):
            fn, name = name, None
            return decorator(fn)
        return decorator

    def _finalize(self, root: Run) -> None:
        spans: List[Run] = []

        def _walk(run: Run) -> None:
            for child in run.children:
                if not child._ended:
                    child.end()
                spans.append(child)
                _walk(child)

        _walk(root)
        payload = {"run": root.to_dict(), "spans": [s.to_dict() for s in spans]}
        with self._lock:
            self.completed.append(payload)
        if self.sqlite_path is not None:
            try:
                self._write_sqlite([root, *spans])
            except Exception as e:
                logger.warning("Trace SQLite write failed: %s", e)
        if self.export_url is not None:
            with self._lock:
                self._buffer.append(payload)
            self.flush()

    def _write_sqlite(self, runs: List[Run]) -> None:
        rows = []
        for run in runs:
            d = run.to_dict()
            rows.append(
                (
                    d["id"],
                    d["project"],
                    d["name"],
                    d["run_type"],
                    d["start_time"],
                    d["end_time"],
                    d["status"],
                    d["error"],
                    d["model"],
                    d["provider"],
                    d["input_chars"],
                    d["output_chars"],
                    d["input_sha256"],
                    d["output_sha256"],
                    json.dumps(d["inputs"]) if d["inputs"] is not None else None,
                    json.dumps(d["outputs"]) if d["outputs"] is not None else None,
                    json.dumps(d["tokens"]) if d["tokens"] is not None else None,
                    d["cost"],
                    json.dumps(d["tags"]),
                    d["parent_id"],
                )
            )
        with self._lock, sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany(_INSERT, rows)

    @property
    def pending_exports(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> int:
        """POST buffered traces to the platform; failures stay buffered."""
        if self.export_url is None:
            return 0
        with self._lock:
            pending = list(self._buffer)
        if not pending:
            return 0
        url = f"{self.export_url}/api/v1/runs"
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        sent: List[Dict[str, Any]] = []
        for payload in pending:
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}")
            except Exception as e:
                logger.warning(
                    "Trace export to %s failed (%s); %d run(s) kept buffered for retry",
                    url,
                    e,
                    len(pending) - len(sent),
                )
                break
            sent.append(payload)
        if sent:
            with self._lock:
                self._buffer = [p for p in self._buffer if not any(p is s for s in sent)]
        return len(sent)


class TracedModel:
    """Drop-in tracing wrapper around any BaseLLM-compatible model.

    Intercepts async ``generate``/``chat``/``generate_stream``/``chat_stream``
    and sync ``generate_sync``/``chat_sync``; every other attribute is proxied
    to the wrapped model, so it composes with ``guard()`` and
    ``track_costs()`` wrappers. Each call records one llm run with latency
    (start/end times), token usage (real when exposed, else the chars/4
    estimate), and cost from the wrapped model's pricing.
    """

    def __init__(self, model: Any, tracer: Optional[RunTracer] = None, name: Optional[str] = None):
        self._model = model
        self.tracer = tracer if tracer is not None else get_default_tracer()
        self._name = name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _start(self, method: str, inputs: Any) -> Run:
        base = self._name or getattr(self._model, "model_name", type(self._model).__name__)
        return self.tracer.start_run(
            f"{base}.{method}",
            "llm",
            inputs=inputs,
            model=getattr(self._model, "model_name", None),
            provider=getattr(self._model, "PROVIDER_NAME", type(self._model).__name__),
        )

    def _finish(self, run: Run, input_text: str, output_text: str, result: Any = None) -> None:
        usage = _extract_usage(result, self._model)
        if usage is not None:
            input_tokens, output_tokens = usage
            estimated = False
        else:
            input_tokens = estimate_tokens(input_text)
            output_tokens = estimate_tokens(output_text)
            estimated = True
        cost, unpriced = _resolve_cost(self._model, input_tokens, output_tokens)
        run.end(
            outputs=output_text,
            tokens={"input": input_tokens, "output": output_tokens, "estimated": estimated},
            cost=None if unpriced else cost,
        )

    @staticmethod
    def _output_text(result: Any) -> str:
        return result if isinstance(result, str) else str(result)

    async def generate(self, prompt: str, **kwargs) -> Any:
        run = self._start("generate", prompt)
        try:
            result = await self._model.generate(prompt, **kwargs)
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, prompt, self._output_text(result), result)
        return result

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        run = self._start("chat", messages)
        try:
            result = await self._model.chat(messages, **kwargs)
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, _messages_text(messages), self._output_text(result), result)
        return result

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        run = self._start("generate_stream", prompt)
        chunks: List[str] = []
        try:
            async for chunk in self._model.generate_stream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, prompt, "".join(chunks))

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        run = self._start("chat_stream", messages)
        chunks: List[str] = []
        try:
            async for chunk in self._model.chat_stream(messages, **kwargs):
                chunks.append(chunk)
                yield chunk
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, _messages_text(messages), "".join(chunks))

    def generate_sync(self, prompt: str, **kwargs) -> Any:
        run = self._start("generate_sync", prompt)
        try:
            result = self._model.generate_sync(prompt, **kwargs)
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, prompt, self._output_text(result), result)
        return result

    def chat_sync(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        run = self._start("chat_sync", messages)
        try:
            result = self._model.chat_sync(messages, **kwargs)
        except Exception as e:
            run.end(error=f"{type(e).__name__}: {e}")
            raise
        self._finish(run, _messages_text(messages), self._output_text(result), result)
        return result


_default_tracer: Optional[RunTracer] = None
_default_tracer_lock = threading.Lock()


def get_default_tracer() -> RunTracer:
    """Process-wide tracer configured from MULTIMIND_* env vars on first use."""
    global _default_tracer
    with _default_tracer_lock:
        if _default_tracer is None:
            _default_tracer = RunTracer(
                project=os.environ.get("MULTIMIND_TRACING_PROJECT", "default"),
                export_url=os.environ.get("MULTIMIND_PLATFORM_URL") or None,
                api_key=os.environ.get("MULTIMIND_PLATFORM_KEY") or None,
                sqlite_path=os.environ.get("MULTIMIND_TRACING_DB") or None,
            )
        return _default_tracer


def reset_default_tracer() -> None:
    global _default_tracer
    with _default_tracer_lock:
        _default_tracer = None


def trace_model(
    model: Any, tracer: Optional[RunTracer] = None, name: Optional[str] = None
) -> TracedModel:
    """Convenience wrapper: ``trace_model(model) -> TracedModel`` on the default tracer."""
    return TracedModel(model, tracer=tracer, name=name)
