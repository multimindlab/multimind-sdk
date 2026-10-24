"""REST routes that expose SDK components which previously had no HTTP surface.

Every route here is a thin call into the SDK class it names; nothing is
reimplemented. Missing optional dependencies become a ``200`` capability
report on ``GET`` routes and a ``501`` with an install hint on ``POST``
routes, never a ``500``. ``GET /v1/features`` is the machine-readable
registry the dashboard and the feature-chain test read.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from ..core.chat import chat_manager
from .models import get_model_handler

logger = logging.getLogger(__name__)

FINETUNE_INSTALL = "pip install 'multimind-sdk[finetune]'"


# ---------------------------------------------------------------------------
# Model resolution shared by every model-backed route
# ---------------------------------------------------------------------------


class _HandlerLLM:
    """BaseLLM-shaped view over a gateway ``ModelHandler`` (string in, string out)."""

    def __init__(self, handler: Any, provider: str):
        self._handler = handler
        self.PROVIDER_NAME = provider
        self.model_name = getattr(getattr(handler, "config", None), "model_name", provider)
        client = getattr(handler, "_client", None)
        if client is not None:
            # Lets Agent use native tool calling when the client is OpenAI-shaped.
            self.client = client

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self._handler.generate(prompt, **kwargs)
        return str(getattr(response, "content", response))

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self._handler.chat(messages, **kwargs)
        return str(getattr(response, "content", response))

    async def embeddings(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError(f"The {self.PROVIDER_NAME} gateway handler has no embeddings")


def _resolve_llm(model: str) -> Any:
    """Resolve a ``/v1/chat`` model name to a BaseLLM-compatible object.

    ``SDKModelHandler`` wraps a real ``multimind.models`` instance, which is
    returned directly; the legacy handlers (openai, anthropic, ollama,
    huggingface) are adapted so ``generate``/``chat`` return strings.
    """
    try:
        handler = get_model_handler(model)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"Model {model} needs an extra: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # provider client refused to construct (no key, bad base URL)
        raise HTTPException(status_code=400, detail=f"Model {model} is not available: {exc}")
    inner = getattr(handler, "_llm", None)
    if inner is not None:
        return inner
    return _HandlerLLM(handler, model)


def _not_installed(exc: Exception, install: str) -> HTTPException:
    return HTTPException(status_code=501, detail=f"{exc}. Install with: {install}")


# ---------------------------------------------------------------------------
# Evaluation: hallucination detection
# ---------------------------------------------------------------------------

evaluation_router = APIRouter(prefix="/v1/evaluation", tags=["evaluation"])


class HallucinationRequest(BaseModel):
    response: str = Field(..., description="Model output to check")
    sources: List[str] = Field(..., description="Source texts the response must be grounded in")
    judge_model: Optional[str] = Field(
        default=None,
        description="Optional /v1/chat model name; adds an LLM-judge claim check",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Paris is the capital of France. It has 80 million people.",
                    "sources": ["Paris is the capital and largest city of France."],
                }
            ]
        }
    }


class SentenceVerdict(BaseModel):
    sentence: str
    verdict: str
    score: float
    best_source_snippet: Optional[str] = None


class HallucinationResponse(BaseModel):
    score: float
    unsupported_count: int
    summary: str
    sentences: List[SentenceVerdict]
    claims: Optional[Dict[str, Any]] = Field(
        default=None, description="Claim-level verdicts; present only when judge_model is set"
    )


@evaluation_router.post("/hallucination", response_model=HallucinationResponse)
async def hallucination_check(request: HallucinationRequest):
    """Grounding check via ``HallucinationDetector.check_grounding``; with
    ``judge_model`` it also runs ``check_claims`` through that model."""
    from ..evaluation.hallucination import HallucinationDetector

    detector = HallucinationDetector()
    report = detector.check_grounding(request.response, request.sources)
    claims = None
    if request.judge_model:
        judge = _resolve_llm(request.judge_model)
        try:
            claim_report = await detector.check_claims(request.response, request.sources, judge)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"Judge output unusable: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Judge model call failed: {exc}")
        claims = {
            "score": claim_report.score,
            "summary": claim_report.summary(),
            "claims": [{"claim": c.claim, "verdict": c.verdict} for c in claim_report.claims],
        }
    return HallucinationResponse(
        score=report.score,
        unsupported_count=report.unsupported_count,
        summary=report.summary(),
        sentences=[SentenceVerdict(**vars(s)) for s in report.sentences],
        claims=claims,
    )


# ---------------------------------------------------------------------------
# Sessions: model switching with ModelSession semantics
# ---------------------------------------------------------------------------

sessions_router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


class SessionSwitchRequest(BaseModel):
    model: str = Field(..., description="Target /v1/chat model name")
    strategy: str = Field(default="full", description="full | summary | adapted")

    model_config = {"json_schema_extra": {"examples": [{"model": "claude", "strategy": "summary"}]}}


class _NamedModel:
    """Proxy giving a resolved model the provider name ModelSession reports."""

    def __init__(self, inner: Any, name: str):
        self._inner = inner
        self.model_name = name

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


@sessions_router.post("/{session_id}/switch", responses={404: {"description": "Session not found"}})
async def switch_session_model(session_id: str, request: SessionSwitchRequest):
    """Move a gateway chat session to another model with ``ModelSession.switch``.

    The gateway session is a plain record (``ChatSession``), so this builds a
    ``ModelSession`` seeded with its history, calls ``switch`` with the
    requested strategy, writes the resulting history back, re-targets
    ``session.model`` and appends the transfer entry to ``session.metadata``.
    ``summary`` calls the session's current model to compact the history;
    ``full`` and ``adapted`` make no model call.
    """
    from ..client.model_session import _STRATEGIES, ModelSession

    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.strategy not in _STRATEGIES:
        raise HTTPException(
            status_code=400, detail=f"Unknown strategy {request.strategy!r}; use {_STRATEGIES}"
        )

    target = _NamedModel(_resolve_llm(request.model), request.model)
    if request.strategy == "summary":
        source: Any = _NamedModel(_resolve_llm(session.model), session.model)
    else:
        source = SimpleNamespace(model_name=session.model)

    model_session = ModelSession(source, system_prompt=session.system_prompt)
    model_session._history = [{"role": m.role, "content": m.content} for m in session.messages]
    try:
        await model_session.switch(target, strategy=request.strategy)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model request failed during switch: {exc}")

    transfer = model_session.transfers[-1]
    if request.strategy != "full":
        session.messages = []
        for message in model_session.history:
            session.add_message(
                role=message["role"], content=message["content"], model=request.model
            )
    session.model = request.model
    session.metadata.setdefault("transfers", []).append(transfer)
    return {
        "session_id": session.session_id,
        "model": session.model,
        "strategy_applied": transfer["strategy"],
        "transfer": transfer,
        "message_count": len(session.messages),
    }


# ---------------------------------------------------------------------------
# Context transfer
# ---------------------------------------------------------------------------

context_router = APIRouter(prefix="/v1/context-transfer", tags=["context-transfer"])


class ContextTransferRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="role/content message dicts")
    source_model: str = Field(..., description="Provider the conversation came from")
    target_model: str = Field(..., description="Provider the prompt is formatted for")
    strategy: str = Field(
        default="concise", description="Summary style: concise | detailed | structured"
    )
    last_n: int = Field(default=5, ge=1, description="Recent turns to keep")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "Plan a 3-day trip to Rome."},
                        {"role": "assistant", "content": "Day 1: Colosseum..."},
                    ],
                    "source_model": "chatgpt",
                    "target_model": "claude",
                    "strategy": "concise",
                    "last_n": 5,
                }
            ]
        }
    }


@context_router.post("")
async def transfer_context(request: ContextTransferRequest):
    """``ContextTransferAPI.transfer_context_api`` over ``ContextTransferManager``:
    extract the last N turns, summarize, and format for the target provider."""
    from ..context_transfer.api import ContextTransferAPI

    if request.strategy not in ("concise", "detailed", "structured"):
        raise HTTPException(
            status_code=400, detail="strategy must be concise, detailed or structured"
        )
    result = ContextTransferAPI().transfer_context_api(
        request.source_model,
        request.target_model,
        request.messages,
        options={"last_n": request.last_n, "summary_type": request.strategy},
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "context transfer failed"))
    return result


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

agents_router = APIRouter(prefix="/v1/agents", tags=["agents"])

_AGENT_TOOLS = {"calculator": "multimind.agents.tools.calculator:CalculatorTool"}


class AgentRunRequest(BaseModel):
    model: str = Field(..., description="/v1/chat model name")
    task: str = Field(..., description="Task for the agent")
    tools: List[str] = Field(default_factory=lambda: ["calculator"], description="Tool names")
    max_turns: int = Field(default=5, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [{"model": "openai", "task": "What is 17 * 23?", "tools": ["calculator"]}]
        }
    }


def _load_tool(name: str) -> Any:
    spec = _AGENT_TOOLS.get(name)
    if spec is None:
        raise HTTPException(
            status_code=400, detail=f"Unknown tool {name!r}; available: {sorted(_AGENT_TOOLS)}"
        )
    module_path, cls_name = spec.split(":")
    return getattr(importlib.import_module(module_path), cls_name)()


@agents_router.post("/run")
async def run_agent(request: AgentRunRequest):
    """``multimind.agents.Agent.run`` with the requested tools. Returns the
    final answer and the tool-call trace (native function calling when the
    model client is OpenAI-shaped, keyword routing otherwise)."""
    from ..agents.agent import Agent, _supports_native_tools

    model = _resolve_llm(request.model)
    tools = [_load_tool(name) for name in request.tools]
    agent = Agent(model=model, tools=tools, max_turns=request.max_turns)
    result = await agent.run(request.task)
    tool_calls = result.get("tool_calls", [])
    if result.get("type") == "tool":
        tool_calls = [{k: v for k, v in result.items() if k != "type"}]
    return {
        "answer": result.get("result"),
        "error": result.get("error"),
        "native_tools": _supports_native_tools(model),
        "tool_calls": jsonable_encoder(tool_calls),
        "tools": [tool.name for tool in tools],
    }


# ---------------------------------------------------------------------------
# Memory demo store
# ---------------------------------------------------------------------------

memory_router = APIRouter(prefix="/v1/memory", tags=["memory"])

# kind -> (module, class, needs_model)
_MEMORY_KINDS = {
    "buffer": ("multimind.memory.buffer", "BufferMemory", False),
    "summary": ("multimind.memory.summary", "SummaryMemory", True),
    "episodic": ("multimind.memory.episodic", "EpisodicMemory", True),
    "semantic": ("multimind.memory.semantic", "SemanticMemory", True),
    "procedural": ("multimind.memory.procedural", "ProceduralMemory", True),
}
_MEMORY_STORE: Dict[str, Any] = {}


class _NullLLM:
    """Placeholder used only to prove a memory class constructs; raises if used."""

    model_name = "null"

    async def generate(self, *args, **kwargs):
        raise RuntimeError("no model configured for this memory")

    async def chat(self, *args, **kwargs):
        raise RuntimeError("no model configured for this memory")

    async def embeddings(self, *args, **kwargs):
        raise RuntimeError("no model configured for this memory")


class MemoryMessage(BaseModel):
    role: str = Field(default="user")
    content: str
    model: Optional[str] = Field(
        default=None, description="/v1/chat model name; required for model-backed kinds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"role": "user", "content": "Remember that my meeting is at 3pm."}]
        }
    }


def _memory_class(kind: str):
    if kind not in _MEMORY_KINDS:
        raise HTTPException(
            status_code=404, detail=f"Unknown memory kind; use {sorted(_MEMORY_KINDS)}"
        )
    module_path, cls_name, needs_model = _MEMORY_KINDS[kind]
    return getattr(importlib.import_module(module_path), cls_name), needs_model


def _construct_memory(kind: str, llm: Any):
    cls, needs_model = _memory_class(kind)
    if kind == "buffer":
        return cls(enable_backup=False)
    return cls(llm=llm)


def reset_memory_store() -> None:
    _MEMORY_STORE.clear()


@memory_router.get("")
async def list_memory_kinds():
    """Every demo kind with its class, whether it constructs here, and whether
    this process currently holds an instance."""
    kinds = []
    for kind, (module_path, cls_name, needs_model) in _MEMORY_KINDS.items():
        entry: Dict[str, Any] = {
            "kind": kind,
            "module": module_path,
            "class": cls_name,
            "needs_model": needs_model,
            "active": kind in _MEMORY_STORE,
        }
        try:
            _construct_memory(kind, _NullLLM())
            entry["constructible"] = True
        except Exception as exc:
            entry["constructible"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        kinds.append(entry)
    return {"kinds": kinds}


@memory_router.post("/{kind}/messages")
async def add_memory_message(kind: str, message: MemoryMessage):
    """``add_message`` on the real memory class; the instance is created on
    first use and lives for the process."""
    _, needs_model = _memory_class(kind)
    memory = _MEMORY_STORE.get(kind)
    if memory is None:
        llm = None
        if needs_model:
            if not message.model:
                raise HTTPException(status_code=400, detail=f"{kind} memory needs a model")
            llm = _resolve_llm(message.model)
        memory = _construct_memory(kind, llm)
        _MEMORY_STORE[kind] = memory
    try:
        await memory.add_message({"role": message.role, "content": message.content})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{kind} memory rejected the message: {exc}")
    messages = await memory.get_messages()
    return {"kind": kind, "stored": True, "message_count": len(messages)}


@memory_router.get("/{kind}/messages")
async def get_memory_messages(kind: str):
    """``get_messages`` on the stored instance; empty when none was created."""
    _memory_class(kind)
    memory = _MEMORY_STORE.get(kind)
    messages = await memory.get_messages() if memory is not None else []
    return {"kind": kind, "active": memory is not None, "messages": jsonable_encoder(messages)}


@memory_router.delete("/{kind}")
async def clear_memory(kind: str):
    """``clear`` on the stored instance and drop it."""
    _memory_class(kind)
    memory = _MEMORY_STORE.pop(kind, None)
    if memory is not None:
        await memory.clear()
    return {"kind": kind, "cleared": memory is not None}


# ---------------------------------------------------------------------------
# Vector store inventory
# ---------------------------------------------------------------------------

vector_router = APIRouter(prefix="/v1/vector-stores", tags=["vector-stores"])

_LOCAL_BACKENDS = {
    "FAISSBackend",
    "ChromaBackend",
    "AnnoyBackend",
    "SklearnBackend",
    "SQLiteVSSBackend",
    "LanceDBBackend",
}
_VECTOR_STORE_CACHE: Optional[Dict[str, Any]] = None
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"numpy", "multimind"}


def _client_module(module_path: str) -> Optional[str]:
    """First third-party import in the backend module: its client package."""
    try:
        source = Path(importlib.util.find_spec(module_path).origin).read_text(encoding="utf-8")
    except Exception:
        return None
    for node in ast.walk(ast.parse(source)):
        names: List[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] not in _STDLIB:
                return name
    return None


def _find_spec(name: Optional[str]) -> bool:
    if name is None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _minimal_kwargs(name: str, scratch: str) -> Dict[str, Any]:
    # Dummy credentials; nothing may connect at construction time.
    return {
        "AlibabaCloudOpenSearchBackend": {"api_key": "k", "endpoint": "example.invalid"},
        "AnalyticDBBackend": {"host": "h", "user": "u", "password": "p", "database": "d"},
        "AnnoyBackend": {"vector_dim": 4},
        "AstraDBBackend": {"token": "t", "api_endpoint": "https://example.invalid"},
        "AtlasBackend": {"uri": "mongodb://example.invalid"},
        "AzureCosmosDBBackend": {"endpoint": "https://example.invalid", "key": "k"},
        "AzureSearchBackend": {"endpoint": "https://example.invalid", "api_key": "k"},
        "ChromaBackend": {
            "collection_name": "inventory_probe",
            "chroma_settings": {
                "is_persistent": True,
                "persist_directory": os.path.join(scratch, "chroma"),
                "anonymized_telemetry": False,
            },
        },
        "HippoBackend": {"api_key": "k", "endpoint": "e"},
        "HologresBackend": {
            "host": "h",
            "port": 80,
            "dbname": "d",
            "user": "u",
            "password": "p",
            "vector_dim": 4,
        },
        "LLMRailsBackend": {"api_key": "k", "datastore_id": "d"},
        "LanceDBBackend": {"uri": os.path.join(scratch, "lancedb"), "vector_dim": 4},
        "MatchingEngineBackend": {"project": "p", "index_id": "i", "endpoint_id": "e"},
        "MomentoVectorIndexBackend": {"auth_token": "t"},
        "MongoDBAtlasBackend": {"uri": "mongodb://example.invalid"},
        "NucliaDBBackend": {"key": "k", "kb": "kb"},
        "PineconeBackend": {"api_key": "k", "environment": "env"},
        "RocksetDBBackend": {"api_key": "k"},
        "SQLiteVSSBackend": {"db_path": os.path.join(scratch, "vectors.db"), "dim": 4},
        "SklearnBackend": {"dim": 4},
        "SupabaseVectorStore": {"url": "https://example.invalid", "api_key": "k"},
        "TencentVectorDBVectorStore": {"endpoint": "e", "username": "u", "password": "p"},
        "TigrisVectorStore": {"project": "p"},
        "TileDBVectorStore": {"array_uri": os.path.join(scratch, "tiledb")},
        "TypesenseVectorStore": {"api_key": "k"},
        "USearchVectorStore": {"index_path": os.path.join(scratch, "usearch.idx")},
        "VectaraVectorStore": {"customer_id": "c", "api_key": "k", "corpus_id": "1"},
        "XataVectorStore": {"db_url": "https://example.invalid", "api_key": "k"},
        "ZepVectorStore": {"api_url": "https://example.invalid"},
    }.get(name, {})


def _probe_backend(name: str, module_path: str, experimental: bool, scratch: str) -> Dict[str, Any]:
    from ..vector_store import get_backend_class
    from ..vector_store.base import VectorStoreConfig, VectorStoreType

    client = _client_module(f"multimind.vector_store{module_path}")
    entry: Dict[str, Any] = {
        "name": name,
        "module": f"multimind.vector_store{module_path}",
        "client_module": client,
        "client_installed": _find_spec(client),
        "local": name in _LOCAL_BACKENDS,
        "experimental": experimental,
        "constructible": False,
    }
    if experimental:
        entry["reason"] = "not implemented; every operation raises NotImplementedError"
        return entry
    try:
        cls = get_backend_class(name)
    except Exception as exc:
        entry["reason"] = f"{type(exc).__name__}: {exc}"
        return entry
    if cls is None:
        entry["reason"] = "module import failed (client library missing)"
        return entry
    try:
        if name == "FAISSBackend":
            cls(VectorStoreConfig({"dimension": 4}, VectorStoreType.FAISS))
        else:
            cls(**_minimal_kwargs(name, scratch))
        entry["constructible"] = True
    except Exception as exc:
        entry["reason"] = f"{type(exc).__name__}: {exc}"
    return entry


@vector_router.get("")
async def list_vector_stores(refresh: bool = False):
    """Inventory of every backend in the ``multimind.vector_store`` registry,
    probed once per process: constructible with a minimal config, client
    package installed, local or client-backed, experimental stub."""
    global _VECTOR_STORE_CACHE
    if _VECTOR_STORE_CACHE is not None and not refresh:
        return _VECTOR_STORE_CACHE
    from ..vector_store import _BACKEND_MODULES, _EXPERIMENTAL_MODULES

    scratch = tempfile.mkdtemp(prefix="multimind-vs-")
    backends = [
        _probe_backend(name, path, False, scratch)
        for name, path in sorted(_BACKEND_MODULES.items())
    ] + [
        _probe_backend(name, path, True, scratch)
        for name, path in sorted(_EXPERIMENTAL_MODULES.items())
    ]
    _VECTOR_STORE_CACHE = {
        "total": len(backends),
        "constructible": sum(1 for b in backends if b["constructible"]),
        "backends": backends,
    }
    return _VECTOR_STORE_CACHE


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

finetune_router = APIRouter(prefix="/v1/finetune", tags=["fine-tuning"])


class SyntheticQARequest(BaseModel):
    chunks: List[str] = Field(..., description="Document chunks to generate Q/A pairs from")
    model: str = Field(..., description="/v1/chat model name")
    max_pairs: Optional[int] = Field(default=None, ge=1, description="Cap on returned pairs")
    questions_per_chunk: int = Field(default=3, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunks": [
                        "MultiMind SDK ships a PII guard that masks emails and phone numbers."
                    ],
                    "model": "openai",
                    "max_pairs": 5,
                }
            ]
        }
    }


@finetune_router.post("/synthetic-qa")
async def synthetic_qa(request: SyntheticQARequest):
    """``SyntheticQAGenerator.generate``: grounded Q/A pairs plus their
    Alpaca-style instruction records."""
    from ..fine_tuning.synthetic_data import SyntheticQAGenerator

    generator = SyntheticQAGenerator(
        _resolve_llm(request.model), num_questions_per_chunk=request.questions_per_chunk
    )
    try:
        pairs = await generator.generate(request.chunks)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Model output unusable: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}")
    if request.max_pairs is not None:
        pairs = pairs[: request.max_pairs]
    return {
        "count": len(pairs),
        "pairs": [
            {
                "question": p.question,
                "answer": p.answer,
                "source_chunk": p.source_chunk,
                "grounding_score": p.grounding_score,
            }
            for p in pairs
        ],
        "records": [p.to_instruction_record() for p in pairs],
    }


_FINETUNE_METHODS = {
    "synthetic_qa": ("multimind.fine_tuning.synthetic_data", "SyntheticQAGenerator"),
    "lora": ("multimind.fine_tuning.lora_trainer", "LoRATrainer"),
    "qlora": ("multimind.fine_tuning.qlora_trainer", "QLoraTuner"),
    "adapter": ("multimind.fine_tuning.adapter_tuning", "AdapterTuner"),
    "conversion": ("multimind.model_conversion", "ModelConversionManager"),
}


def _import_report(module_path: str, attr: Optional[str] = None) -> Dict[str, Any]:
    try:
        module = importlib.import_module(module_path)
        if attr:
            getattr(module, attr)
        return {"available": True}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


@finetune_router.get("/capabilities")
async def finetune_capabilities():
    """Which fine-tuning entry points import in this process (torch-gated ones
    say why they do not), with the install hint."""
    methods = {}
    for name, (module_path, attr) in _FINETUNE_METHODS.items():
        methods[name] = {"module": module_path, "class": attr, **_import_report(module_path, attr)}
    return {
        "torch_installed": _find_spec("torch"),
        "install": FINETUNE_INSTALL,
        "methods": methods,
    }


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

convert_router = APIRouter(prefix="/v1/convert", tags=["conversion"])

# Mirrors the choices of ``multimind convert``.
_CONVERT_SOURCES = ["huggingface", "pytorch", "tensorflow", "onnx", "ollama"]
_CONVERT_TARGETS = ["gguf", "safetensors", "tflite", "ort", "onnx"]


@convert_router.get("/formats")
async def conversion_formats():
    """Source and target formats from ``multimind.model_conversion`` when it
    imports (needs torch); otherwise an honest ``available: false`` report."""
    report = _import_report("multimind.model_conversion", "ModelConversionManager")
    if not report["available"]:
        return {"available": False, "install": FINETUNE_INSTALL, "reason": report["reason"]}
    from ..model_conversion import ModelConversionManager

    manager = ModelConversionManager()
    return {
        "available": True,
        "sources": _CONVERT_SOURCES,
        "targets": _CONVERT_TARGETS,
        "converters": manager.list_converters(),
    }


# ---------------------------------------------------------------------------
# Compliance: regulatory watcher and differential privacy
# ---------------------------------------------------------------------------

compliance_router = APIRouter(prefix="/v1/compliance", tags=["compliance"])


class RegulatorySource(BaseModel):
    name: str
    url: Optional[str] = Field(default=None, description="Page to fetch in online mode")
    content: Optional[str] = Field(default=None, description="Page text for offline mode")


class RegulatoryCheckRequest(BaseModel):
    sources: List[RegulatorySource]
    offline: bool = Field(default=True, description="Use supplied content; no network")
    state_path: Optional[str] = Field(
        default=None,
        description="JSON file keeping last-seen hashes (per-process temp file by default)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sources": [{"name": "eu-ai-act", "content": "Article 1 ..."}],
                    "offline": True,
                }
            ]
        }
    }


_REGULATORY_STATE = os.path.join(tempfile.gettempdir(), "multimind_regulatory_state.json")


@compliance_router.post("/regulatory-check")
async def regulatory_check(request: RegulatoryCheckRequest):
    """``RegulatoryWatcher``: hash each source and report ``ChangeEvent``s.
    Offline mode runs ``check_offline`` on the supplied content; online mode
    runs ``check`` and fetches every ``url``."""
    from ..compliance.regulatory_watch import RegulatoryWatcher

    sources = []
    for source in request.sources:
        if request.offline and source.content is None:
            raise HTTPException(
                status_code=400, detail=f"offline mode needs content for source {source.name!r}"
            )
        if not request.offline and not source.url:
            raise HTTPException(
                status_code=400, detail=f"online mode needs url for source {source.name!r}"
            )
        sources.append({"name": source.name, "url": source.url or f"offline://{source.name}"})

    watcher = RegulatoryWatcher(sources, state_path=request.state_path or _REGULATORY_STATE)
    if request.offline:
        with tempfile.TemporaryDirectory(prefix="multimind-reg-") as tmp:
            paths = {}
            for index, source in enumerate(request.sources):
                path = Path(tmp) / f"{index}.txt"
                path.write_text(source.content or "", encoding="utf-8")
                paths[source.name] = str(path)
            events = watcher.check_offline(paths)
    else:
        try:
            events = await watcher.check()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Fetch failed: {exc}")
    return {
        "mode": "offline" if request.offline else "online",
        "state_path": str(watcher.state_path),
        "events": [vars(event) for event in events],
    }


class PrivatizeRequest(BaseModel):
    value: Optional[float] = Field(default=None, description="One numeric value")
    values: Optional[List[float]] = Field(default=None, description="Several numeric values")
    epsilon: float = Field(..., gt=0, description="Privacy parameter spent by this call")
    sensitivity: float = Field(default=1.0, gt=0)
    budget: Optional[float] = Field(default=None, gt=0, description="Total epsilon budget")
    budget_id: Optional[str] = Field(
        default=None, description="Name a budget to carry it across calls in this process"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"value": 42, "epsilon": 0.5, "budget": 1.0, "budget_id": "demo"}]
        }
    }


_BUDGETS: Dict[str, Any] = {}


def reset_privacy_budgets() -> None:
    _BUDGETS.clear()


@compliance_router.post("/dp/privatize")
async def privatize(request: PrivatizeRequest):
    """``DPMechanism.privatize`` with optional ``PrivacyBudget`` accounting;
    400 with the SDK message once the budget is exhausted."""
    from ..compliance.advanced import BudgetExhaustedError, DPMechanism, PrivacyBudget

    if request.value is None and request.values is None:
        raise HTTPException(status_code=400, detail="Provide value or values")
    budget = None
    if request.budget is not None:
        if request.budget_id:
            budget = _BUDGETS.get(request.budget_id)
            if budget is None:
                budget = PrivacyBudget(request.budget)
                _BUDGETS[request.budget_id] = budget
        else:
            budget = PrivacyBudget(request.budget)
    mechanism = DPMechanism(request.epsilon, sensitivity=request.sensitivity, budget=budget)
    data: Any = request.values if request.values is not None else request.value
    try:
        noised = mechanism.privatize(data)
    except BudgetExhaustedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    body: Dict[str, Any] = {"epsilon": request.epsilon, "sensitivity": request.sensitivity}
    if request.values is not None:
        body["values"] = noised
    else:
        body["value"] = noised
    if budget is not None:
        body["budget"] = {
            "id": request.budget_id,
            "total_epsilon": budget.total_epsilon,
            "spent_epsilon": budget.spent_epsilon,
            "remaining_epsilon": budget.remaining_epsilon,
        }
    return body


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

mcp_router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


class McpInvokeRequest(BaseModel):
    arguments: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {"examples": [{"arguments": {"text": "Mail me at jane@example.com"}}]}
    }


def _mcp_tools() -> Dict[str, Any]:
    from ..mcp_server import server

    return {tool.__name__: tool for tool in server._TOOLS}


@mcp_router.get("/tools")
async def list_mcp_tools():
    """The tool functions ``multimind.mcp_server`` registers, read without
    starting the server; ``server_startable`` says whether the ``mcp`` package is here."""
    tools = []
    for name, fn in _mcp_tools().items():
        parameters = {
            p.name: {
                "required": p.default is inspect.Parameter.empty,
                "annotation": str(p.annotation).replace("typing.", ""),
            }
            for p in inspect.signature(fn).parameters.values()
        }
        tools.append(
            {"name": name, "description": (fn.__doc__ or "").strip(), "parameters": parameters}
        )
    return {
        "server": "multimind-compliance",
        "server_startable": _find_spec("mcp"),
        "install": "pip install 'multimind-sdk[mcp]'",
        "tools": tools,
    }


@mcp_router.post("/tools/{name}")
async def invoke_mcp_tool(name: str, request: McpInvokeRequest):
    """Call one MCP tool function in-process (``audit_log`` appends to
    ``MULTIMIND_AUDIT_LOG``)."""
    tool = _mcp_tools().get(name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Unknown MCP tool {name!r}")
    try:
        result = tool(**request.arguments)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Bad arguments for {name}: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{name} failed: {exc}")
    return {"tool": name, "result": jsonable_encoder(result)}


# ---------------------------------------------------------------------------
# Framework integrations
# ---------------------------------------------------------------------------

integrations_router = APIRouter(prefix="/v1/integrations", tags=["integrations"])

# framework -> (adapter module, importable client module)
_FRAMEWORKS = {
    "langchain": ("multimind.integrations.frameworks.langchain", "langchain_core"),
    "llamaindex": ("multimind.integrations.frameworks.llamaindex", "llama_index.core"),
    "crewai": ("multimind.integrations.frameworks.crewai", "crewai"),
    "autogen": ("multimind.integrations.frameworks.autogen", "autogen_core"),
    "haystack": ("multimind.integrations.frameworks.haystack", "haystack"),
    "openai_client": ("multimind.integrations.frameworks.openai_client", "openai"),
}


@integrations_router.get("")
async def list_integrations():
    """Each adapter in ``multimind.integrations.frameworks``: whether the
    framework is installed and the adapter entry points it exports."""
    from ..integrations import frameworks

    out = []
    for name, (module_path, client) in _FRAMEWORKS.items():
        entries = sorted(
            export for export, (path, _dist) in frameworks._EXPORTS.items() if path == module_path
        )
        dists = sorted(
            {dist for _e, (path, dist) in frameworks._EXPORTS.items() if path == module_path}
        )
        out.append(
            {
                "framework": name,
                "module": module_path,
                "client_module": client,
                "installed": _find_spec(client),
                "install": f"pip install {' '.join(dists)}",
                "entry_points": entries,
            }
        )
    return {"frameworks": out}


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

traces_router = APIRouter(prefix="/v1/traces", tags=["tracing"])


class _DemoStubModel:
    """Deterministic stand-in so a trace can be recorded without a provider key."""

    model_name = "demo-stub"
    PROVIDER_NAME = "stub"

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"stub reply to {len(prompt)} characters"


class TraceDemoRequest(BaseModel):
    prompt: str = Field(default="Hello from the trace demo")


@traces_router.get("")
async def list_traces(limit: int = 20):
    """Recent root runs from the ``RunTracer`` SQLite sink at ``MULTIMIND_TRACING_DB``.
    Hashes and lengths only, as the tracer records by default."""
    db = os.environ.get("MULTIMIND_TRACING_DB")
    if not db:
        return {
            "available": False,
            "reason": "MULTIMIND_TRACING_DB is not set; the default tracer has no SQLite sink",
            "runs": [],
        }
    if not Path(db).exists():
        return {"available": True, "db": db, "runs": [], "reason": "no runs recorded yet"}
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM runs WHERE parent_id IS NULL ORDER BY start_time DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    finally:
        conn.close()
    return {"available": True, "db": db, "runs": [dict(row) for row in rows]}


@traces_router.post("/demo")
async def trace_demo(request: TraceDemoRequest):
    """One ``trace_model`` call on a stub model through the default tracer so
    the UI can show a run; persisted when ``MULTIMIND_TRACING_DB`` is set."""
    from ..observability.tracing import get_default_tracer, trace_model

    tracer = get_default_tracer()
    traced = trace_model(_DemoStubModel(), tracer=tracer)
    await traced.generate(request.prompt)
    payload = tracer.completed[-1]
    return {
        "persisted": tracer.sqlite_path is not None,
        "db": str(tracer.sqlite_path) if tracer.sqlite_path else None,
        "run": payload["run"],
        "spans": payload["spans"],
    }


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

features_router = APIRouter(prefix="/v1/features", tags=["system"])

# (name, module, attr-or-None, routes). ``available`` is computed by import.
_FEATURES: List[Dict[str, Any]] = [
    {
        "name": "chat",
        "module": "multimind.gateway.models",
        "routes": ["POST /v1/chat", "POST /v1/generate"],
    },
    {"name": "compare", "module": "multimind.gateway.models", "routes": ["POST /v1/compare"]},
    {
        "name": "models",
        "module": "multimind.gateway.monitoring",
        "routes": ["GET /v1/models", "GET /v1/metrics", "POST /v1/health/check"],
    },
    {
        "name": "sessions",
        "module": "multimind.core.chat",
        "routes": [
            "POST /v1/sessions",
            "GET /v1/sessions",
            "GET /v1/sessions/{session_id}",
            "DELETE /v1/sessions/{session_id}",
            "POST /v1/sessions/{session_id}/messages",
            "POST /v1/sessions/{session_id}/save",
            "POST /v1/sessions/load",
        ],
    },
    {
        "name": "session-switch",
        "module": "multimind.client.model_session",
        "routes": ["POST /v1/sessions/{session_id}/switch"],
    },
    {
        "name": "compliance-monitor",
        "module": "multimind.gateway.compliance_api",
        "routes": [
            "POST /v1/compliance/monitor",
            "POST /v1/compliance/report",
            "GET /v1/compliance/regulations",
            "GET /v1/compliance/healthcare/use-cases",
            "GET /v1/compliance/dashboard",
            "GET /v1/compliance/alerts",
            "POST /v1/compliance/alerts/configure",
        ],
    },
    {
        "name": "guard-proxy",
        "module": "multimind.gateway.guard_proxy",
        "routes": [
            "POST /v1/chat/completions",
            "POST /v1/embeddings",
            "POST /v1/guardrails/reload",
        ],
    },
    {
        "name": "scan-text",
        "module": "multimind.compliance.guard",
        "routes": ["POST /api/scan-text"],
    },
    {
        "name": "audit",
        "module": "multimind.compliance.guard",
        "routes": ["GET /api/audit", "GET /api/summary"],
    },
    {
        "name": "evidence",
        "module": "multimind.compliance.reporting",
        "routes": ["GET /api/evidence-report"],
    },
    {
        "name": "inventory",
        "module": "multimind.observability.ai_inventory",
        "routes": ["GET /api/inventory"],
    },
    {
        "name": "costs",
        "module": "multimind.observability.cost_tracker",
        "routes": ["GET /api/costs"],
    },
    {
        "name": "guardrails",
        "module": "multimind.dashboard.server",
        "routes": ["GET /api/guardrails", "PUT /api/guardrails"],
    },
    {
        "name": "rag",
        "module": "multimind.gateway.rag_api",
        "routes": [
            "POST /documents",
            "DELETE /documents",
            "GET /documents/count",
            "POST /files",
            "POST /query",
            "POST /generate",
            "POST /models/switch",
            "POST /token",
        ],
    },
    {
        "name": "hallucination",
        "module": "multimind.evaluation.hallucination",
        "routes": ["POST /v1/evaluation/hallucination"],
    },
    {
        "name": "context-transfer",
        "module": "multimind.context_transfer",
        "routes": ["POST /v1/context-transfer"],
    },
    {"name": "agents", "module": "multimind.agents", "routes": ["POST /v1/agents/run"]},
    {
        "name": "memory",
        "module": "multimind.memory.buffer",
        "routes": [
            "GET /v1/memory",
            "POST /v1/memory/{kind}/messages",
            "GET /v1/memory/{kind}/messages",
            "DELETE /v1/memory/{kind}",
        ],
    },
    {
        "name": "vector-stores",
        "module": "multimind.vector_store",
        "routes": ["GET /v1/vector-stores"],
    },
    {
        "name": "synthetic-qa",
        "module": "multimind.fine_tuning.synthetic_data",
        "routes": ["POST /v1/finetune/synthetic-qa"],
    },
    {
        "name": "fine-tuning",
        "module": "multimind.fine_tuning.lora_trainer",
        "routes": ["GET /v1/finetune/capabilities"],
        "install": FINETUNE_INSTALL,
    },
    {
        "name": "model-conversion",
        "module": "multimind.model_conversion",
        "routes": ["GET /v1/convert/formats"],
        "install": FINETUNE_INSTALL,
    },
    {
        "name": "regulatory-watch",
        "module": "multimind.compliance.regulatory_watch",
        "routes": ["POST /v1/compliance/regulatory-check"],
    },
    {
        "name": "differential-privacy",
        "module": "multimind.compliance.advanced",
        "routes": ["POST /v1/compliance/dp/privatize"],
    },
    {
        "name": "mcp",
        "module": "multimind.mcp_server",
        "routes": ["GET /v1/mcp/tools", "POST /v1/mcp/tools/{name}"],
    },
    {
        "name": "integrations",
        "module": "multimind.integrations.frameworks",
        "routes": ["GET /v1/integrations"],
    },
    {
        "name": "tracing",
        "module": "multimind.observability.tracing",
        "routes": ["GET /v1/traces", "POST /v1/traces/demo"],
    },
]


@features_router.get("")
async def list_features():
    """Machine-readable registry: every feature with its SDK module, routes,
    and ``available`` computed by importing the module in this process."""
    features = []
    for feature in _FEATURES:
        entry = {k: v for k, v in feature.items() if k != "install"}
        report = _import_report(feature["module"])
        entry["available"] = report["available"]
        if not report["available"]:
            entry["reason"] = report["reason"]
            if feature.get("install"):
                entry["install"] = feature["install"]
        features.append(entry)
    return {"count": len(features), "features": features}


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------

ROUTERS = [
    evaluation_router,
    sessions_router,
    context_router,
    agents_router,
    memory_router,
    vector_router,
    finetune_router,
    convert_router,
    compliance_router,
    mcp_router,
    integrations_router,
    traces_router,
    features_router,
]

OPENAPI_TAGS = [
    {"name": "evaluation", "description": "Hallucination and grounding checks"},
    {"name": "context-transfer", "description": "Reformat a conversation for another provider"},
    {"name": "agents", "description": "Tool-using agents"},
    {"name": "memory", "description": "Per-process memory demo store"},
    {"name": "vector-stores", "description": "Vector store backend inventory"},
    {"name": "fine-tuning", "description": "Synthetic data and fine-tuning capabilities"},
    {"name": "conversion", "description": "Model conversion capability report"},
    {"name": "mcp", "description": "MCP compliance tools in-process"},
    {"name": "integrations", "description": "Framework adapter availability"},
    {"name": "tracing", "description": "Run tracer sink and demo"},
]


def init_app(app: FastAPI) -> None:
    """Include every feature router and register their OpenAPI tags."""
    for router in ROUTERS:
        app.include_router(router)
    existing = {tag["name"] for tag in (app.openapi_tags or [])}
    app.openapi_tags = list(app.openapi_tags or []) + [
        tag for tag in OPENAPI_TAGS if tag["name"] not in existing
    ]
