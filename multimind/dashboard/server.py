"""Self-hosted AI governance dashboard over local MultiMind artifacts.

Read-only web UI for compliance officers and team leads: PII audit trail
(:class:`multimind.compliance.guard.AuditLog` JSONL), spend and chargeback
(:mod:`multimind.observability.cost_tracker` JSONL), shadow-AI inventory
(:func:`multimind.observability.ai_inventory.scan_project`), plus a no-code
guardrails config file the guard proxy loads via ``multimind serve --config``.
All data comes from local files; the server makes no external calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .. import __version__
from ..compliance.guard import _LABELS, PIIDetector
from ..compliance.reporting import build_evidence_report
from ..config_store import PROVIDERS, get_config_store, is_secret_key
from ..observability.ai_inventory import scan_project
from ..observability.cost_tracker import CostTracker, load_tracker

PII_TYPES = tuple(_LABELS)
_STRATEGIES = ("mask", "hash", "remove")
_REPORT_FORMATS = {"md": "text/markdown", "html": "text/html"}
_MAX_SCAN_CHARS = 1_000_000

_STATIC_DIR = Path(__file__).parent / "static"


class DashboardSettings(BaseModel):
    """Dashboard configuration; build from env with :meth:`from_env`."""

    audit_log: str = "audit.jsonl"
    costs_log: str = "costs.jsonl"
    project_path: str = "."
    guardrails_path: str = "guardrails.json"
    host: str = "127.0.0.1"
    port: int = 8501
    write_api_key: Optional[str] = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "DashboardSettings":
        """Read settings from MULTIMIND_DASHBOARD_* env vars; overrides win."""
        env_map = {
            "audit_log": "MULTIMIND_DASHBOARD_AUDIT_LOG",
            "costs_log": "MULTIMIND_DASHBOARD_COSTS_LOG",
            "project_path": "MULTIMIND_DASHBOARD_PROJECT",
            "guardrails_path": "MULTIMIND_DASHBOARD_GUARDRAILS",
            "host": "MULTIMIND_DASHBOARD_HOST",
            "port": "MULTIMIND_DASHBOARD_PORT",
            "write_api_key": "MULTIMIND_DASHBOARD_API_KEY",
        }
        values: Dict[str, Any] = {}
        for field_name, env_var in env_map.items():
            raw = os.getenv(env_var)
            if raw:
                values[field_name] = raw
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


class GuardrailsConfig(BaseModel):
    """Schema for the guardrails file authored in the dashboard UI."""

    strategy: str = "mask"
    block_on: List[str] = Field(default_factory=list)
    budget_max_cost: Optional[float] = None
    scan_output: bool = True

    model_config = {"extra": "forbid"}

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: str) -> str:
        if v not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {v!r} (use one of {_STRATEGIES})")
        return v

    @field_validator("block_on")
    @classmethod
    def _check_block_on(cls, v: List[str]) -> List[str]:
        unknown = sorted(set(v) - set(PII_TYPES))
        if unknown:
            raise ValueError(f"Unknown PII types {unknown} (use a subset of {sorted(PII_TYPES)})")
        return list(dict.fromkeys(v))

    @field_validator("budget_max_cost")
    @classmethod
    def _check_budget(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("budget_max_cost must be positive")
        return v


class ScanTextRequest(BaseModel):
    """Ad-hoc PII scan request; the text is scanned in memory, never logged."""

    text: str = Field(min_length=1, max_length=_MAX_SCAN_CHARS)
    strategy: Optional[str] = None

    model_config = {"extra": "forbid"}

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _STRATEGIES:
            raise ValueError(f"Unknown redaction strategy: {v!r} (use one of {_STRATEGIES})")
        return v


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                records.append(entry)
    return records


def _parse_ts(value: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _audit_stats(records: List[Dict[str, Any]], since: Optional[datetime] = None) -> Dict[str, Any]:
    events = blocked = detections = 0
    by_type: Counter = Counter()
    for r in records:
        if since is not None:
            ts = _parse_ts(r.get("timestamp"))
            if ts is None or ts < since:
                continue
        events += 1
        if r.get("blocked"):
            blocked += 1
        pii = r.get("pii_types")
        if isinstance(pii, dict):
            for ptype, count in pii.items():
                try:
                    by_type[ptype] += int(count)
                except (TypeError, ValueError):
                    continue
        try:
            detections += int(r.get("count", 0))
        except (TypeError, ValueError):
            pass
    return {
        "events": events,
        "pii_detections": detections,
        "blocked": blocked,
        "by_type": dict(by_type),
    }


def _load_costs(path: str) -> Optional[CostTracker]:
    if not Path(path).is_file():
        return None
    try:
        return load_tracker(path)
    except (OSError, ValueError):
        return None


def _daily_spend(tracker: CostTracker) -> List[Dict[str, Any]]:
    days: Dict[str, Dict[str, Any]] = {}
    for r in tracker.records:
        day = str(r.timestamp)[:10]
        g = days.setdefault(day, {"cost": 0.0, "calls": 0})
        g["cost"] += r.cost
        g["calls"] += 1
    return [{"date": day, **g} for day, g in sorted(days.items())]


def _read_guardrails(path: str) -> GuardrailsConfig:
    p = Path(path)
    if not p.is_file():
        return GuardrailsConfig()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return GuardrailsConfig(**data)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not parse guardrails file {path}: {exc}"
        ) from exc


def _budget_status(guardrails: GuardrailsConfig, spent: float) -> Optional[Dict[str, Any]]:
    max_cost = guardrails.budget_max_cost
    if not max_cost:
        return None
    # Compared against all logged spend; the proxy Budget itself is per session.
    return {
        "max_cost": max_cost,
        "spent": spent,
        "remaining": max(0.0, max_cost - spent),
        "exceeded": spent >= max_cost,
        "basis": "all logged spend vs configured ceiling",
    }


class TestProviderRequest(BaseModel):
    provider: str

    model_config = {"extra": "forbid"}

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        name = v.strip().lower()
        if name not in PROVIDERS:
            raise ValueError(f"Unknown provider {v!r} (use one of {', '.join(PROVIDERS)})")
        return name


def _scrub_secrets(text: str) -> str:
    """Replace any env credential value that leaks into an error message."""
    for name, value in os.environ.items():
        if value and len(value) >= 8 and is_secret_key(name) and value in text:
            text = text.replace(value, "***")
    return text


def _effective_config(guardrails: GuardrailsConfig) -> Dict[str, Any]:
    from ..core.config import config

    store = get_config_store()
    providers: Dict[str, Any] = {}
    for name in PROVIDERS:
        model_config = config.get_model_config(name)
        providers[name] = {
            "model_name": model_config.model_name,
            "base_url": model_config.api_base,
            "enabled": config.is_provider_enabled(name),
            "has_api_key": bool(model_config.api_key),
        }
    env_origins = (
        os.getenv("MULTIMIND_CORS_ORIGINS") or os.getenv("MULTIMIND_ALLOWED_ORIGINS") or ""
    )
    cors = store.get("cors_origins")
    if cors is None:
        cors = [o.strip() for o in env_origins.split(",") if o.strip()]
    return {
        "path": str(store.path),
        "exists": store.exists(),
        "overrides": store.as_dict(),
        "providers_known": list(PROVIDERS),
        "pii_types": list(PII_TYPES),
        "effective": {
            "default_model": config.effective_default_model(),
            "log_level": config.effective_log_level(),
            "cors_origins": cors,
            "proxy": {
                "strategy": store.get("proxy.strategy", guardrails.strategy),
                "block_on": store.get("proxy.block_on", guardrails.block_on),
                "budget": store.get("proxy.budget", guardrails.budget_max_cost),
                "scan_output": store.get("proxy.scan_output", guardrails.scan_output),
            },
            "providers": providers,
        },
    }


def create_dashboard_app(settings: Optional[DashboardSettings] = None) -> FastAPI:
    """Build the governance dashboard FastAPI app (env-derived if omitted)."""
    settings = settings or DashboardSettings.from_env()

    app = FastAPI(
        title="MultiMind Governance Dashboard",
        description=(
            "Local, mostly-read-only dashboard over MultiMind governance artifacts: "
            "PII audit trail, spend and chargeback, shadow-AI inventory, and "
            "no-code guardrail authoring, and runtime settings (/api/config; API keys "
            "stay in env). Read endpoints are unauthenticated; the write endpoints "
            "(PUT /api/guardrails, PUT /api/config) are "
            "unauthenticated by default but can be gated with an X-API-Key header "
            "by setting MULTIMIND_DASHBOARD_API_KEY. Bind to localhost or put "
            "behind a reverse proxy."
        ),
        version=__version__,
        openapi_tags=[
            {"name": "governance", "description": "Audit, costs, inventory, guardrails"},
            {"name": "system", "description": "Health and readiness probes"},
        ],
    )
    app.state.settings = settings
    detector = PIIDetector()
    inventory_cache: Dict[str, Any] = {}
    inventory_lock = threading.Lock()

    def _cached_inventory(refresh: bool = False) -> Dict[str, Any]:
        with inventory_lock:
            if refresh or "report" not in inventory_cache:
                inventory_cache["report"] = scan_project(settings.project_path).to_dict()
                inventory_cache["scanned_at"] = datetime.now(timezone.utc).isoformat()
            return dict(inventory_cache)

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "healthy", "version": __version__}

    @app.get("/ready", tags=["system"])
    async def readiness_check():
        return {"status": "ready", "version": __version__}

    @app.get("/api/summary", tags=["governance"])
    async def summary():
        records = _read_jsonl(settings.audit_log)
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        tracker = _load_costs(settings.costs_log)
        total_cost = tracker.total_cost if tracker else 0.0
        cost_records = tracker.records if tracker else []
        last_24h_cost = sum(
            r.cost for r in cost_records if (ts := _parse_ts(r.timestamp)) and ts >= since
        )
        guardrails = _read_guardrails(settings.guardrails_path)
        return {
            "audit": {
                "available": Path(settings.audit_log).is_file(),
                "all_time": _audit_stats(records),
                "last_24h": _audit_stats(records, since=since),
            },
            "costs": {
                "available": tracker is not None,
                "total_cost": total_cost,
                "last_24h_cost": last_24h_cost,
                "calls": len(cost_records),
                "models": sorted({r.model for r in cost_records}),
            },
            "budget": _budget_status(guardrails, total_cost),
        }

    @app.get("/api/audit", tags=["governance"])
    async def audit(
        limit: int = 50,
        offset: int = 0,
        type: Optional[str] = None,
        blocked: Optional[bool] = None,
    ):
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        records = list(reversed(_read_jsonl(settings.audit_log)))  # newest first
        if type is not None:
            records = [r for r in records if type in (r.get("pii_types") or {})]
        if blocked is not None:
            records = [r for r in records if bool(r.get("blocked")) is blocked]
        return {
            "total": len(records),
            "limit": limit,
            "offset": offset,
            "pii_types": list(PII_TYPES),
            "records": records[offset : offset + limit],
        }

    @app.get("/api/costs", tags=["governance"])
    async def costs(period: Optional[str] = None):
        """Spend summary; ``period`` is an ISO timestamp prefix (``multimind audit costs --period``)."""
        tracker = _load_costs(settings.costs_log)
        if tracker is None:
            return {
                "available": False,
                "chargeback": {"period": period, "calls": 0, "total_cost": 0.0, "by_tag": {}},
                "by_model": {},
                "daily": [],
            }
        return {
            "available": True,
            "chargeback": tracker.chargeback(period=period),
            "by_model": tracker.by_model(),
            "daily": _daily_spend(tracker),
        }

    @app.get("/api/inventory", tags=["governance"])
    async def inventory(refresh: bool = False, include_env: bool = False):
        """Shadow-AI inventory; ``include_env`` mirrors ``multimind audit scan --include-env``."""
        try:
            if include_env:
                # Env-key names are per-request, so bypass the cache like the CLI does.
                report = scan_project(settings.project_path, include_env=True).to_dict()
                return {
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                    "cached": False,
                    **report,
                }
            cached = _cached_inventory(refresh)
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "scanned_at": cached["scanned_at"],
            "cached": not refresh,
            **cached["report"],
        }

    @app.post("/api/scan-text", tags=["governance"])
    async def scan_text(request: ScanTextRequest):
        # Echoes findings back to the caller only; nothing is logged server-side.
        if request.strategy is not None:
            redacted, matches = detector.redact(request.text, request.strategy)
        else:
            redacted, matches = None, detector.detect(request.text)
        return {
            "findings": [
                {"type": m.type, "start": m.start, "end": m.end, "text": m.text} for m in matches
            ],
            "counts": dict(Counter(m.type for m in matches)),
            "total": len(matches),
            "redacted": redacted,
        }

    @app.get("/api/evidence-report", tags=["governance"])
    async def evidence_report(
        format: str = "html",
        period: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        costs_log: Optional[str] = None,
    ):
        """Evidence report; params mirror ``multimind compliance report-evidence``."""
        if format not in _REPORT_FORMATS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown format: {format!r} (use one of {sorted(_REPORT_FORMATS)})",
            )
        if project is not None:
            if not Path(project).is_dir():
                raise HTTPException(
                    status_code=400, detail=f"project is not a directory: {project}"
                )
            inventory = scan_project(project).to_dict()
        else:
            try:
                inventory = _cached_inventory()["report"]
            except NotADirectoryError:
                inventory = None
        if costs_log is not None:
            if not Path(costs_log).is_file():
                raise HTTPException(status_code=400, detail=f"costs_log not found: {costs_log}")
            costs_path: Optional[str] = costs_log
        else:
            costs_path = settings.costs_log if Path(settings.costs_log).is_file() else None
        report = build_evidence_report(
            audit_log=settings.audit_log if Path(settings.audit_log).is_file() else None,
            costs_log=costs_path,
            inventory=inventory,
            period=period,
            organization=organization,
        )
        body = report.to_markdown() if format == "md" else report.to_html()
        return Response(
            content=body,
            media_type=_REPORT_FORMATS[format],
            headers={
                "Content-Disposition": f'attachment; filename="evidence-report.{format}"',
            },
        )

    @app.get("/api/guardrails", tags=["governance"])
    async def get_guardrails():
        return {
            "path": settings.guardrails_path,
            "exists": Path(settings.guardrails_path).is_file(),
            "pii_types": list(PII_TYPES),
            "config": _read_guardrails(settings.guardrails_path).model_dump(),
        }

    @app.put("/api/guardrails", tags=["governance"])
    async def put_guardrails(
        config: GuardrailsConfig, x_api_key: Optional[str] = Header(default=None)
    ):
        if settings.write_api_key and x_api_key != settings.write_api_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
        path = Path(settings.guardrails_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.model_dump(), indent=2) + "\n", encoding="utf-8")
        return {
            "path": settings.guardrails_path,
            "exists": True,
            "config": config.model_dump(),
            "apply_hint": (
                f"POST /v1/guardrails/reload on the API, or "
                f"multimind serve --config {settings.guardrails_path}"
            ),
        }

    @app.get("/api/config", tags=["governance"])
    async def get_config():
        """Effective runtime settings; API keys are reported only as ``has_api_key``."""
        return _effective_config(_read_guardrails(settings.guardrails_path))

    @app.put("/api/config", tags=["governance"])
    async def put_config(values: Dict[str, Any], x_api_key: Optional[str] = Header(default=None)):
        """Partial update of the config store (flat dotted or nested keys); ``null`` clears a key."""
        if settings.write_api_key and x_api_key != settings.write_api_key:
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
        store = get_config_store()
        try:
            store.set_many(values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not write {store.path}: {exc}")
        guardrails = _read_guardrails(settings.guardrails_path)
        if any(str(k).startswith("proxy") for k in values):
            # Mirror proxy settings into the guardrails file so /v1/guardrails/reload applies them.
            defaults = GuardrailsConfig().model_dump()
            merged = guardrails.model_dump()
            for field in ("strategy", "block_on", "budget", "scan_output"):
                target = "budget_max_cost" if field == "budget" else field
                merged[target] = store.get(f"proxy.{field}", defaults[target])
            guardrails = GuardrailsConfig(**merged)
            path = Path(settings.guardrails_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(guardrails.model_dump(), indent=2) + "\n", encoding="utf-8")
        body = _effective_config(guardrails)
        body["apply_hint"] = (
            "Model, log level and provider settings apply to the next request. "
            "Proxy settings are written to the guardrails file; POST /v1/guardrails/reload "
            "to apply them. CORS origins apply on the next start."
        )
        return body

    @app.post("/api/config/test-provider", tags=["governance"])
    async def test_provider(request: TestProviderRequest):
        """One short real chat call through the provider; returns ok/latency, never keys."""
        from ..gateway import models as gateway_models

        started = time.perf_counter()
        try:
            handler = await asyncio.wait_for(
                asyncio.to_thread(gateway_models.get_model_handler, request.provider), timeout=10
            )
            await asyncio.wait_for(
                handler.chat([{"role": "user", "content": "ping"}], max_tokens=5), timeout=10
            )
        except asyncio.TimeoutError:
            latency = int((time.perf_counter() - started) * 1000)
            return {"ok": False, "latency_ms": latency, "error": "timed out after 10s"}
        except Exception as exc:  # provider errors are reported, not raised
            latency = int((time.perf_counter() - started) * 1000)
            return {"ok": False, "latency_ms": latency, "error": _scrub_secrets(str(exc))[:500]}
        return {
            "ok": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": None,
        }

    @app.get("/config.js", include_in_schema=False)
    async def dashboard_config():
        base = os.getenv("MULTIMIND_API_BASE", "")
        key = os.getenv("MULTIMIND_API_KEY", "")
        body = (
            f"window.MULTIMIND_API_BASE={json.dumps(base)};\n"
            f"window.MULTIMIND_API_KEY={json.dumps(key)};\n"
        )
        return Response(
            content=body,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


def start(settings: Optional[DashboardSettings] = None) -> None:
    """Start the dashboard with uvicorn (blocking)."""
    import uvicorn

    settings = settings or DashboardSettings.from_env()
    uvicorn.run(create_dashboard_app(settings), host=settings.host, port=settings.port)
