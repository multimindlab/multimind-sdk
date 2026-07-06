"""
FastAPI-based API Gateway for MultiMind
"""

import importlib.util
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import __version__
from ..core.chat import chat_manager
from ..core.config import config
from ..core.models import ModelResponse
from ..core.monitoring import ModelHealth, monitor
from ..gateway.models import get_model_handler
from .compliance_api import init_app as init_compliance_app

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind API Gateway",
    description=(
        "Unified gateway for MultiMind services: chat, generation, model comparison, "
        "chat sessions, monitoring, and compliance. Provider API keys are read from "
        "the environment at request time; requests fail with 503 when no provider is "
        "configured and 400 when the requested model is unavailable."
    ),
    version=__version__,
    openapi_tags=[
        {"name": "models", "description": "Model discovery and availability"},
        {"name": "generation", "description": "Chat, generation, and model comparison"},
        {"name": "sessions", "description": "Persistent chat sessions"},
        {"name": "monitoring", "description": "Metrics and model health"},
        {"name": "compliance", "description": "Compliance monitoring and reporting"},
        {"name": "system", "description": "Health and readiness probes"},
    ],
)


def _get_cors_origins() -> List[str]:
    # CORS is off unless MULTIMIND_CORS_ORIGINS (or legacy MULTIMIND_ALLOWED_ORIGINS)
    # is set to a comma-separated list of origins.
    raw = os.getenv("MULTIMIND_CORS_ORIGINS") or os.getenv("MULTIMIND_ALLOWED_ORIGINS") or ""
    return [o.strip() for o in raw.split(",") if o.strip()]


if _get_cors_origins():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize compliance routes
init_compliance_app(app)


class ErrorResponse(BaseModel):
    detail: str


ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Requested model is not available"},
    422: {"description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
    503: {"model": ErrorResponse, "description": "No models configured"},
}


def _model_status() -> Dict[str, bool]:
    # Availability is evaluated per request so keys added to the environment
    # after startup are picked up without a restart.
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY") or config.openai.api_key),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY") or config.anthropic.api_key),
        "ollama": bool(os.getenv("OLLAMA_API_BASE") or config.ollama.api_base),
        "groq": bool(os.getenv("GROQ_API_KEY") or config.groq.api_key),
        "huggingface": bool(os.getenv("HUGGINGFACE_API_KEY") or config.huggingface.api_key)
        or importlib.util.find_spec("transformers") is not None,
    }


# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Model for chat messages"""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    model: Optional[str] = Field(default=None, description="Model that generated the message")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional message metadata"
    )


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default=config.default_model, description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "model": "openai",
                    "temperature": 0.7,
                }
            ]
        }
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to generate from")
    model: str = Field(default=config.default_model, description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

    model_config = {
        "json_schema_extra": {
            "examples": [{"prompt": "Explain retrieval-augmented generation.", "model": "openai"}]
        }
    }


class CompareRequest(BaseModel):
    """Request model for comparing models"""

    prompt: str = Field(..., description="Prompt to compare models on")
    models: List[str] = Field(
        default=["openai", "anthropic", "ollama"], description="Models to compare"
    )
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

    model_config = {
        "json_schema_extra": {
            "examples": [{"prompt": "Write a limerick.", "models": ["openai", "anthropic"]}]
        }
    }


class CompareResponse(BaseModel):
    responses: Dict[str, ModelResponse]


# New Pydantic models for monitoring and chat
class MetricsResponse(BaseModel):
    """Response model for metrics endpoint"""

    metrics: Dict[str, Any]
    health: Dict[str, ModelHealth]


class SessionCreate(BaseModel):
    """Request model for creating a chat session"""

    model: str
    system_prompt: Optional[str] = None
    metadata: Dict = {}


class SessionResponse(BaseModel):
    """Response model for chat session"""

    session_id: str
    model: str
    created_at: datetime
    updated_at: datetime
    message_count: int


# Privacy Compliance Pydantic models
class DataPurposeRequest(BaseModel):
    purpose_id: str = Field(..., description="Unique identifier for the purpose")
    name: str = Field(..., description="Name of the purpose")
    description: str = Field(..., description="Description of the purpose")
    legal_basis: str = Field(..., description="Legal basis for data processing")
    retention_period: int = Field(..., description="Retention period in days")
    data_categories: List[str] = Field(..., description="List of data categories")


class RiskScoreRequest(BaseModel):
    entity_id: str = Field(..., description="Entity identifier")
    entity_type: str = Field(default="system", description="Type of entity")


class DashboardRequest(BaseModel):
    dashboard_id: str = Field(..., description="Dashboard identifier")
    name: str = Field(..., description="Dashboard name")
    description: str = Field(..., description="Dashboard description")
    refresh_interval: int = Field(default=3600, description="Refresh interval in seconds")


class ReportTemplateRequest(BaseModel):
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    regulation: str = Field(..., description="Regulation name")
    jurisdiction: str = Field(..., description="Jurisdiction")
    sections: List[Dict[str, Any]] = Field(..., description="Report sections")


class TrainingRequest(BaseModel):
    training_id: str = Field(..., description="Training identifier")
    title: str = Field(..., description="Training title")
    description: str = Field(..., description="Training description")
    modules: List[Dict[str, Any]] = Field(..., description="Training modules")
    target_audience: List[str] = Field(..., description="Target audience")
    duration: int = Field(..., description="Duration in minutes")
    completion_criteria: Dict[str, Any] = Field(..., description="Completion criteria")


# Dependency to validate model configuration
async def validate_model_config():
    status = _model_status()
    if not any(status.values()):
        raise HTTPException(
            status_code=503,
            detail="No models are properly configured. Please check your API keys.",
        )
    return status


@app.get("/", tags=["system"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MultiMind API Gateway",
        "version": __version__,
        "models": list(_model_status().keys()),
    }


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint (no auth, no provider keys required)."""
    return {"status": "healthy", "version": __version__}


@app.get("/ready", tags=["system"])
async def readiness_check():
    """Readiness probe; handlers are created lazily per request."""
    return {"status": "ready", "version": __version__}


@app.get("/v1/models", tags=["models"], responses=ERROR_RESPONSES)
async def list_models(status: Dict = Depends(validate_model_config)):
    """List available models and their status"""
    return {
        "models": {
            model: {
                "status": "available" if is_valid else "unavailable",
                "config": {
                    "model_name": config.get_model_config(model).model_name,
                    "temperature": config.get_model_config(model).temperature,
                    "max_tokens": config.get_model_config(model).max_tokens,
                },
            }
            for model, is_valid in status.items()
        }
    }


@app.post("/v1/chat", response_model=ModelResponse, tags=["generation"], responses=ERROR_RESPONSES)
async def chat(request: ChatRequest, status: Dict = Depends(validate_model_config)):
    """Chat with a model"""
    try:
        if request.model not in status or not status[request.model]:
            raise HTTPException(status_code=400, detail=f"Model {request.model} is not available")

        handler = get_model_handler(request.model)
        start_time = time.time()

        try:
            response = await handler.chat(
                [{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Track successful request
            await monitor.track_request(
                model=request.model,
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                cost=0.0,  # Implement cost calculation based on model
                response_time=time.time() - start_time,
                success=True,
            )

            return response

        except Exception as e:
            # Track failed request
            await monitor.track_request(
                model=request.model,
                tokens=0,
                cost=0.0,
                response_time=time.time() - start_time,
                success=False,
                error=str(e),
            )
            raise

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/v1/generate", response_model=ModelResponse, tags=["generation"], responses=ERROR_RESPONSES
)
async def generate(request: GenerateRequest, status: Dict = Depends(validate_model_config)):
    """Generate text from a prompt"""
    try:
        if request.model not in status or not status[request.model]:
            raise HTTPException(status_code=400, detail=f"Model {request.model} is not available")

        handler = get_model_handler(request.model)
        response = await handler.generate(
            request.prompt, temperature=request.temperature, max_tokens=request.max_tokens
        )

        return response

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in generate endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/v1/compare", response_model=CompareResponse, tags=["generation"], responses=ERROR_RESPONSES
)
async def compare(request: CompareRequest, status: Dict = Depends(validate_model_config)):
    """Compare responses from multiple models"""
    try:
        responses = {}
        for model in request.models:
            if model not in status or not status[model]:
                logger.warning(f"Model {model} is not available, skipping")
                continue

            handler = get_model_handler(model)
            response = await handler.generate(
                request.prompt, temperature=request.temperature, max_tokens=request.max_tokens
            )
            responses[model] = response

        return CompareResponse(responses=responses)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error in compare endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/v1/metrics", response_model=MetricsResponse, tags=["monitoring"], responses=ERROR_RESPONSES
)
async def get_metrics(model: Optional[str] = None):
    """Get metrics for models"""
    try:
        metrics = await monitor.get_metrics(model)
        return MetricsResponse(
            metrics=metrics, health={model: health for model, health in monitor.health.items()}
        )
    except Exception:
        logger.exception("Error getting metrics")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/v1/sessions", response_model=SessionResponse, tags=["sessions"], responses=ERROR_RESPONSES
)
async def create_session(request: SessionCreate):
    """Create a new chat session"""
    try:
        session = chat_manager.create_session(
            model=request.model, system_prompt=request.system_prompt, metadata=request.metadata
        )
        return SessionResponse(
            session_id=session.session_id,
            model=session.model,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages),
        )
    except Exception:
        logger.exception("Error creating session")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/v1/sessions",
    response_model=List[SessionResponse],
    tags=["sessions"],
    responses=ERROR_RESPONSES,
)
async def list_sessions():
    """List all chat sessions"""
    try:
        sessions = chat_manager.list_sessions()
        return [SessionResponse(**session) for session in sessions]
    except Exception:
        logger.exception("Error listing sessions")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get(
    "/v1/sessions/{session_id}",
    tags=["sessions"],
    responses={
        **ERROR_RESPONSES,
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session(session_id: str):
    """Get a specific chat session"""
    try:
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "model": session.model,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "model": msg.model,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata,
                }
                for msg in session.messages
            ],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error getting session")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post(
    "/v1/sessions/{session_id}/messages",
    tags=["sessions"],
    responses={
        **ERROR_RESPONSES,
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def add_message(session_id: str, message: ChatMessage, background_tasks: BackgroundTasks):
    """Add a message to a chat session"""
    try:
        session = chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add user message
        session.add_message(
            role=message.role,
            content=message.content,
            model=message.model or session.model,
            metadata=message.metadata,
        )

        # Get model response in background
        async def get_model_response():
            try:
                handler = get_model_handler(session.model)
                response = await handler.chat(
                    [{"role": msg.role, "content": msg.content} for msg in session.messages],
                    temperature=0.7,
                )
                session.add_message(
                    role="assistant",
                    content=response.content,
                    model=session.model,
                    metadata={"usage": response.usage},
                )
            except Exception:
                logger.exception("Error getting model response")
                session.add_message(
                    role="assistant",
                    content="Sorry, I encountered an error while processing your request.",
                    model=session.model,
                    metadata={"error": "Internal server error"},
                )

        background_tasks.add_task(get_model_response)
        return {"status": "message added, processing response"}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error adding message")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete(
    "/v1/sessions/{session_id}",
    tags=["sessions"],
    responses={
        **ERROR_RESPONSES,
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        success = chat_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "session deleted"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error deleting session")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/health/check", tags=["monitoring"], responses=ERROR_RESPONSES)
async def check_health(model: Optional[str] = None):
    """Check health of models"""
    try:
        if model:
            handler = get_model_handler(model)
            health = await monitor.check_health(model, handler)
            return {model: health}
        else:
            health_status = {}
            status = _model_status()
            for model_name, is_available in status.items():
                if is_available:
                    handler = get_model_handler(model_name)
                    health = await monitor.check_health(model_name, handler)
                    health_status[model_name] = health
            return health_status
    except Exception:
        logger.exception("Error checking health")
        raise HTTPException(status_code=500, detail="Internal server error")


class MultiMindAPI:
    """Main API class for MultiMind Gateway"""

    def __init__(self):
        self.app = app

    def configure_routes(self):
        """Configure API routes"""

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}


def start():
    """Start the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()
