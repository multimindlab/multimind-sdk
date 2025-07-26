"""
FastAPI-based API Gateway for MultiMind
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
from datetime import datetime
import uvicorn

from ..core.config import config
from ..core.models import ModelResponse
from ..gateway.models import get_model_handler
from ..core.monitoring import monitor, ModelHealth
from ..core.chat import chat_manager, ChatSession, ChatMessage
from ..compliance.privacy import (
    PrivacyCompliance,
    GovernanceConfig,
    DataCategory,
    NotificationType,
    AuditAction
)
from .compliance_api import init_app as init_compliance_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind API",
    description="API Gateway for MultiMind Services",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize compliance routes
init_compliance_app(app)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    model: Optional[str] = Field(default=None, description="Model that generated the message")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional message metadata")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default=config.default_model, description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to generate from")
    model: str = Field(default=config.default_model, description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class CompareRequest(BaseModel):
    """Request model for comparing models"""
    prompt: str = Field(..., description="Prompt to compare models on")
    models: List[str] = Field(default=["openai", "anthropic", "ollama"], description="Models to compare")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

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
    status = config.validate(value={})
    if not any(status.values()):
        raise HTTPException(
            status_code=500,
            detail="No models are properly configured. Please check your API keys."
        )
    return status

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MultiMind API",
        "version": "1.0.0",
        "models": list(config.validate(value={}).keys())
    }

@app.get("/v1/models")
async def list_models(status: Dict = Depends(validate_model_config)):
    """List available models and their status"""
    return {
        "models": {
            model: {
                "status": "available" if is_valid else "unavailable",
                "config": {
                    "model_name": config.get_model_config(model).model_name,
                    "temperature": config.get_model_config(model).temperature,
                    "max_tokens": config.get_model_config(model).max_tokens
                }
            }
            for model, is_valid in status.items()
        }
    }

@app.post("/v1/chat", response_model=ModelResponse)
async def chat(request: ChatRequest, status: Dict = Depends(validate_model_config)):
    """Chat with a model"""
    try:
        if request.model not in status or not status[request.model]:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} is not available"
            )

        handler = get_model_handler(request.model)
        start_time = time.time()

        try:
            response = await handler.chat(
                [{"role": msg.role, "content": msg.content} for msg in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Track successful request
            await monitor.track_request(
                model=request.model,
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                cost=0.0,  # Implement cost calculation based on model
                response_time=time.time() - start_time,
                success=True
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
                error=str(e)
            )
            raise

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/generate", response_model=ModelResponse)
async def generate(request: GenerateRequest, status: Dict = Depends(validate_model_config)):
    """Generate text from a prompt"""
    try:
        if request.model not in status or not status[request.model]:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} is not available"
            )

        handler = get_model_handler(request.model)
        response = await handler.generate(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return response

    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/compare", response_model=CompareResponse)
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
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            responses[model] = response

        return CompareResponse(responses=responses)

    except Exception as e:
        logger.error(f"Error in compare endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics(model: Optional[str] = None):
    """Get metrics for models"""
    try:
        metrics = await monitor.get_metrics(model)
        return MetricsResponse(
            metrics=metrics,
            health={model: health for model, health in monitor.health.items()}
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new chat session"""
    try:
        session = await chat_manager.create_session(
            model=request.model,
            system_prompt=request.system_prompt,
            metadata=request.metadata
        )
        return SessionResponse(
            session_id=session.session_id,
            model=session.model,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=len(session.messages)
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/sessions", response_model=List[SessionResponse])
async def list_sessions():
    """List all chat sessions"""
    try:
        sessions = await chat_manager.list_sessions()
        return [
            SessionResponse(
                session_id=session.session_id,
                model=session.model,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=len(session.messages)
            )
            for session in sessions
        ]
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session"""
    try:
        session = await chat_manager.get_session(session_id)
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
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/sessions/{session_id}/messages")
async def add_message(
    session_id: str,
    message: ChatMessage,
    background_tasks: BackgroundTasks
):
    """Add a message to a chat session"""
    try:
        session = await chat_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add user message
        await session.add_message(
            role=message.role,
            content=message.content,
            model=message.model,
            metadata=message.metadata
        )

        # Get model response in background
        async def get_model_response():
            try:
                handler = get_model_handler(session.model)
                response = await handler.chat(
                    [{"role": msg.role, "content": msg.content} for msg in session.messages],
                    temperature=0.7
                )
                await session.add_message(
                    role="assistant",
                    content=response.content,
                    model=session.model,
                    metadata={"usage": response.usage}
                )
            except Exception as e:
                logger.error(f"Error getting model response: {str(e)}")
                await session.add_message(
                    role="assistant",
                    content="Sorry, I encountered an error while processing your request.",
                    model=session.model,
                    metadata={"error": str(e)}
                )

        background_tasks.add_task(get_model_response)
        return {"status": "message added, processing response"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        success = await chat_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "session deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/health/check")
async def check_health(model: Optional[str] = None):
    """Check health of models"""
    try:
        if model:
            handler = get_model_handler(model)
            health = await monitor.check_health(model, handler)
            return {model: health}
        else:
            health_status = {}
            for model_name in config.validate(value={}).keys():
                if config.validate(value={})[model_name]:
                    handler = get_model_handler(model_name)
                    health = await monitor.check_health(model_name, handler)
                    health_status[model_name] = health
            return health_status
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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