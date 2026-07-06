"""OpenAI-compatible compliance proxy: PII redaction, budgets, audit trail.

Point any OpenAI client's ``base_url`` at this proxy and every request gets
PII redaction (``PIIDetector``), spend ceilings (``Budget``/``CostTracker``),
and a JSONL audit trail (``AuditLog`` — types/counts/hash tags, never raw
content) with zero code changes. Requests are forwarded to a configurable
upstream (any OpenAI-compatible endpoint, or a named provider) preserving the
OpenAI wire format, including streaming SSE.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator

from .. import __version__
from ..compliance.guard import _LABELS, AuditLog, PIIDetector, _hash_tag
from ..observability.cost_tracker import (
    Budget,
    BudgetExceededError,
    CostTracker,
    estimate_tokens,
)

logger = logging.getLogger(__name__)

_STRATEGIES = ("mask", "hash", "remove")

# Fallback upstream table, used when the model classes (and their BASE_URL /
# API_KEY_ENV_VARS attrs) cannot be imported on a minimal install.
_FALLBACK_UPSTREAMS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "mistral": ("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "deepseek": ("https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
}

_OPENAI_BASE_URL = "https://api.openai.com/v1"


class ProxySettings(BaseModel):
    """Guard proxy configuration; build from env with :meth:`from_env`."""

    upstream: str = "openai"
    upstream_base_url: Optional[str] = None
    upstream_api_key: Optional[str] = None
    strategy: str = "mask"
    block_on: Tuple[str, ...] = ()
    audit_log: Optional[str] = None
    budget: Optional[float] = None
    cost_per_token: Optional[float] = None
    scan_output: bool = True
    stream_overlap: int = 64
    timeout: float = 120.0
    host: str = "127.0.0.1"
    port: int = 8400

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        if v not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {v!r} (use one of {_STRATEGIES})")
        return v

    @field_validator("block_on", mode="before")
    @classmethod
    def _parse_block_on(cls, v: Any) -> Tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            return tuple(t.strip() for t in v.split(",") if t.strip())
        return tuple(v)

    @classmethod
    def from_env(cls, **overrides: Any) -> "ProxySettings":
        """Read settings from MULTIMIND_* env vars; keyword overrides win."""
        values: Dict[str, Any] = {}
        env_map = {
            "upstream": "MULTIMIND_PROXY_UPSTREAM",
            "upstream_base_url": "MULTIMIND_UPSTREAM_BASE_URL",
            "upstream_api_key": "MULTIMIND_UPSTREAM_API_KEY",
            "strategy": "MULTIMIND_PROXY_STRATEGY",
            "block_on": "MULTIMIND_PROXY_BLOCK_ON",
            "audit_log": "MULTIMIND_PROXY_AUDIT_LOG",
            "budget": "MULTIMIND_PROXY_BUDGET",
            "cost_per_token": "MULTIMIND_PROXY_COST_PER_TOKEN",
            "scan_output": "MULTIMIND_PROXY_SCAN_OUTPUT",
            "host": "MULTIMIND_PROXY_HOST",
            "port": "MULTIMIND_PROXY_PORT",
        }
        for field_name, env_var in env_map.items():
            raw = os.getenv(env_var)
            if raw is not None and raw != "":
                if field_name == "scan_output":
                    values[field_name] = raw.strip().lower() not in ("0", "false", "no", "off")
                else:
                    values[field_name] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def _provider_table() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    try:
        from ..models.deepseek import DeepSeekModel
        from ..models.gemini import GeminiModel
        from ..models.groq import GroqModel
        from ..models.mistral import MistralAIModel
        from ..models.openai import OpenAIModel
    except ImportError:
        return dict(_FALLBACK_UPSTREAMS)
    classes = {
        "openai": OpenAIModel,
        "groq": GroqModel,
        "mistral": MistralAIModel,
        "gemini": GeminiModel,
        "deepseek": DeepSeekModel,
    }
    return {
        name: (cls.BASE_URL or _OPENAI_BASE_URL, cls.API_KEY_ENV_VARS)
        for name, cls in classes.items()
    }


def resolve_upstream(settings: ProxySettings) -> Tuple[str, Optional[str]]:
    """Resolve ``(base_url, api_key)`` for the configured upstream."""
    if settings.upstream_base_url:
        return settings.upstream_base_url.rstrip("/"), settings.upstream_api_key
    name = settings.upstream
    if name == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        return f"{host}/v1", settings.upstream_api_key
    table = _provider_table()
    if name not in table:
        known = sorted(table) + ["ollama"]
        raise ValueError(
            f"Unknown upstream {name!r}: use one of {known} or set MULTIMIND_UPSTREAM_BASE_URL"
        )
    base_url, env_vars = table[name]
    api_key = settings.upstream_api_key
    if not api_key:
        api_key = next((os.getenv(v) for v in env_vars if os.getenv(v)), None)
    return base_url.rstrip("/"), api_key


@dataclass
class ProxyState:
    settings: ProxySettings
    base_url: str
    api_key: Optional[str]
    provider: str
    detector: PIIDetector
    audit: Optional[AuditLog]
    tracker: CostTracker
    budget: Optional[Budget]


def _async_client(state: ProxyState) -> httpx.AsyncClient:
    # Module-level factory so tests can monkeypatch in a MockTransport client.
    return httpx.AsyncClient(base_url=state.base_url, timeout=state.settings.timeout)


def _upstream_headers(state: ProxyState, request: Optional[Request] = None) -> Dict[str, str]:
    if state.api_key:
        return {"Authorization": f"Bearer {state.api_key}"}
    if request is not None:
        incoming = request.headers.get("authorization")
        if incoming:
            return {"Authorization": incoming}
    return {}


def _error(status: int, message: str, err_type: str, code: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code, **extra}},
    )


def _message_texts(messages: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return texts


def _redact_messages(state: ProxyState, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    strategy = state.settings.strategy
    redacted: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            redacted.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            redacted.append({**msg, "content": state.detector.redact(content, strategy)[0]})
        elif isinstance(content, list):
            parts = [
                {**part, "text": state.detector.redact(part["text"], strategy)[0]}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content
            ]
            redacted.append({**msg, "content": parts})
        else:
            redacted.append(msg)
    return redacted


def _audit(
    state: ProxyState, direction: str, endpoint: str, matches: List[Any], blocked: bool = False
) -> None:
    # Same record shape as ComplianceGuard._audit: types/counts/hash tags only.
    if state.audit is None:
        return
    types = Counter(m.type for m in matches)
    tags = {f"[{_LABELS.get(m.type, m.type.upper())}:{_hash_tag(m.text)}]" for m in matches}
    state.audit.write(
        {
            "direction": direction,
            "endpoint": endpoint,
            "pii_types": dict(types),
            "count": len(matches),
            "strategy": state.settings.strategy,
            "blocked": blocked,
            "tags": sorted(tags),
        }
    )


def _check_blocked(state: ProxyState, endpoint: str, matches: List[Any]) -> Optional[JSONResponse]:
    blocked = sorted({m.type for m in matches if m.type in state.settings.block_on})
    if not blocked:
        return None
    _audit(state, "input", endpoint, matches, blocked=True)
    return _error(
        400,
        f"Request blocked by compliance policy: found PII types {', '.join(blocked)}",
        "compliance_violation",
        "pii_blocked",
        blocked_types=blocked,
    )


def _check_budget(state: ProxyState) -> Optional[JSONResponse]:
    if state.budget is None:
        return None
    try:
        state.budget.check()
    except BudgetExceededError as exc:
        return _error(
            429,
            str(exc),
            "budget_exceeded",
            "budget_exceeded",
            spent=exc.spent,
            max_cost=exc.max_cost,
        )
    return None


def _cost_for(
    state: ProxyState, model: str, input_tokens: int, output_tokens: int
) -> Tuple[float, bool]:
    """Blended per-token cost; ``(cost, unpriced)`` — never a fabricated price."""
    total = input_tokens + output_tokens
    if state.settings.cost_per_token is not None:
        return total * state.settings.cost_per_token, False
    try:
        from ..models.openai import OpenAIModel

        pricing = OpenAIModel.MODEL_PRICING
    except ImportError:
        pricing = {}
    for prefix in sorted(pricing, key=len, reverse=True):
        if model.startswith(prefix):
            return total * pricing[prefix], False
    return 0.0, True


def _record_cost(
    state: ProxyState,
    endpoint: str,
    model: str,
    usage: Any,
    input_text: str,
    output_text: str,
) -> None:
    input_tokens = output_tokens = None
    if isinstance(usage, dict):
        it = usage.get("prompt_tokens", usage.get("input_tokens"))
        ot = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(it, (int, float)) or isinstance(ot, (int, float)):
            input_tokens = int(it or 0)
            output_tokens = int(ot or 0)
    estimated = input_tokens is None
    if estimated:
        input_tokens = estimate_tokens(input_text)
        output_tokens = estimate_tokens(output_text)
    cost, unpriced = _cost_for(state, model, input_tokens, output_tokens)
    state.tracker.record(
        provider=state.provider,
        model=model or "unknown",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        method=endpoint,
        estimated=estimated,
        unpriced=unpriced,
    )
    if state.budget is not None:
        state.budget.add(cost)


def _passthrough(resp: httpx.Response) -> Response:
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _content_chunk(template: Dict[str, Any], text: str) -> str:
    obj = copy.deepcopy(template)
    obj["choices"][0]["delta"] = {"content": text}
    obj["choices"][0]["finish_reason"] = None
    return _sse(obj)


async def _stream_chat(
    state: ProxyState,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    input_text: str,
) -> AsyncGenerator[str, None]:
    """Forward a streaming chat completion, scanning output across chunk
    boundaries with the same overlap-window scheme as ComplianceGuard."""
    settings = state.settings
    scan = settings.scan_output
    buffer = ""
    full_output = ""
    template: Optional[Dict[str, Any]] = None
    out_matches: List[Any] = []
    model = payload.get("model", "")

    def _flush() -> Optional[str]:
        nonlocal buffer
        if not buffer or template is None:
            buffer = ""
            return None
        segment, matches = state.detector.redact(buffer, settings.strategy)
        out_matches.extend(matches)
        buffer = ""
        return _content_chunk(template, segment)

    client = _async_client(state)
    try:
        async with client.stream(
            "POST", "/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", errors="replace")
                yield _sse(
                    {
                        "error": {
                            "message": f"Upstream error {resp.status_code}: {detail}",
                            "type": "upstream_error",
                            "code": "upstream_error",
                        }
                    }
                )
                yield "data: [DONE]\n\n"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    chunk = _flush()
                    if chunk is not None:
                        yield chunk
                    yield "data: [DONE]\n\n"
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    yield f"data: {data}\n\n"
                    continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta") or {} if choices else {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    full_output += content
                if not scan:
                    yield _sse(obj)
                    continue
                if not isinstance(content, str) or not content:
                    # Flush held-back text before finish/usage chunks pass through.
                    if not choices or choices[0].get("finish_reason") or obj.get("usage"):
                        chunk = _flush()
                        if chunk is not None:
                            yield chunk
                    yield _sse(obj)
                    continue
                template = obj
                buffer += content
                emit_upto = len(buffer) - settings.stream_overlap
                if emit_upto <= 0:
                    continue
                for m in state.detector.detect(buffer):
                    if m.start < emit_upto < m.end:
                        emit_upto = m.start
                if emit_upto <= 0:
                    continue
                segment, buffer = buffer[:emit_upto], buffer[emit_upto:]
                segment, matches = state.detector.redact(segment, settings.strategy)
                out_matches.extend(matches)
                yield _content_chunk(template, segment)
    except httpx.HTTPError as exc:
        yield _sse(
            {
                "error": {
                    "message": f"Could not reach upstream {state.base_url}: {exc}",
                    "type": "upstream_error",
                    "code": "upstream_unreachable",
                }
            }
        )
        yield "data: [DONE]\n\n"
        return
    finally:
        await client.aclose()
    if scan:
        _audit(state, "output", "chat.completions", out_matches)
    _record_cost(state, "chat.completions.stream", model, None, input_text, full_output)


def create_app(settings: Optional[ProxySettings] = None) -> FastAPI:
    """Build the guard proxy FastAPI app from settings (env-derived if omitted)."""
    settings = settings or ProxySettings.from_env()
    base_url, api_key = resolve_upstream(settings)
    state = ProxyState(
        settings=settings,
        base_url=base_url,
        api_key=api_key,
        provider="custom" if settings.upstream_base_url else settings.upstream,
        detector=PIIDetector(),
        audit=AuditLog(settings.audit_log) if settings.audit_log else None,
        tracker=CostTracker(),
        budget=Budget(settings.budget) if settings.budget else None,
    )

    app = FastAPI(
        title="MultiMind Guard Proxy",
        description=(
            "OpenAI-compatible compliance proxy: PII redaction, budget enforcement, "
            "and an audit trail in front of any OpenAI-compatible upstream."
        ),
        version=__version__,
    )
    app.state.proxy = state

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "healthy", "version": __version__}

    @app.get("/ready", tags=["system"])
    async def readiness_check():
        return {"status": "ready", "version": __version__, "upstream": state.base_url}

    @app.get("/v1/models", tags=["models"])
    async def list_models(request: Request):
        client = _async_client(state)
        try:
            resp = await client.get("/models", headers=_upstream_headers(state, request))
        except httpx.HTTPError as exc:
            return _error(
                502,
                f"Could not reach upstream {state.base_url}: {exc}",
                "upstream_error",
                "upstream_unreachable",
            )
        finally:
            await client.aclose()
        return _passthrough(resp)

    @app.post("/v1/chat/completions", tags=["proxy"])
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages") or []
        texts = _message_texts(messages)
        matches = [m for text in texts for m in state.detector.detect(text)]
        blocked = _check_blocked(state, "chat.completions", matches)
        if blocked is not None:
            return blocked
        over_budget = _check_budget(state)
        if over_budget is not None:
            return over_budget
        payload["messages"] = _redact_messages(state, messages)
        _audit(state, "input", "chat.completions", matches)
        input_text = "\n".join(_message_texts(payload["messages"]))
        headers = _upstream_headers(state, request)

        if payload.get("stream"):
            return StreamingResponse(
                _stream_chat(state, payload, headers, input_text),
                media_type="text/event-stream",
            )

        client = _async_client(state)
        try:
            resp = await client.post("/chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            return _error(
                502,
                f"Could not reach upstream {state.base_url}: {exc}",
                "upstream_error",
                "upstream_unreachable",
            )
        finally:
            await client.aclose()
        if resp.status_code != 200:
            return _passthrough(resp)
        body = resp.json()
        output_texts: List[str] = []
        out_matches: List[Any] = []
        for choice in body.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                content = message["content"]
                output_texts.append(content)
                if state.settings.scan_output:
                    redacted, mm = state.detector.redact(content, state.settings.strategy)
                    message["content"] = redacted
                    out_matches.extend(mm)
        if state.settings.scan_output:
            _audit(state, "output", "chat.completions", out_matches)
        _record_cost(
            state,
            "chat.completions",
            payload.get("model", ""),
            body.get("usage"),
            input_text,
            "\n".join(output_texts),
        )
        return JSONResponse(body)

    @app.post("/v1/embeddings", tags=["proxy"])
    async def embeddings(request: Request):
        payload = await request.json()
        raw_input = payload.get("input")
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list):
            texts = [t for t in raw_input if isinstance(t, str)]
        else:
            texts = []
        matches = [m for text in texts for m in state.detector.detect(text)]
        blocked = _check_blocked(state, "embeddings", matches)
        if blocked is not None:
            return blocked
        over_budget = _check_budget(state)
        if over_budget is not None:
            return over_budget
        strategy = state.settings.strategy
        if isinstance(raw_input, str):
            payload["input"] = state.detector.redact(raw_input, strategy)[0]
        elif isinstance(raw_input, list):
            payload["input"] = [
                state.detector.redact(t, strategy)[0] if isinstance(t, str) else t
                for t in raw_input
            ]
        _audit(state, "input", "embeddings", matches)

        client = _async_client(state)
        try:
            resp = await client.post(
                "/embeddings", json=payload, headers=_upstream_headers(state, request)
            )
        except httpx.HTTPError as exc:
            return _error(
                502,
                f"Could not reach upstream {state.base_url}: {exc}",
                "upstream_error",
                "upstream_unreachable",
            )
        finally:
            await client.aclose()
        if resp.status_code != 200:
            return _passthrough(resp)
        body = resp.json()
        _record_cost(
            state,
            "embeddings",
            payload.get("model", ""),
            body.get("usage"),
            "\n".join(texts),
            "",
        )
        return JSONResponse(body)

    return app


def start(settings: Optional[ProxySettings] = None) -> None:
    """Start the guard proxy with uvicorn (blocking)."""
    import uvicorn

    settings = settings or ProxySettings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
