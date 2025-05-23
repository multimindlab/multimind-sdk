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

from .config import config
from .models import ModelResponse, get_model_handler
from .monitoring import monitor, ModelHealth
from .chat import chat_manager, ChatSession, ChatMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind Gateway API",
    description="Unified API Gateway for multiple AI models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ModelResponse(BaseModel):
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None

class CompareResponse(BaseModel):
    responses: Dict[str, ModelResponse]

# New Pydantic models for monitoring and cha
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
        "name": "MultiMind Gateway API",
        "version": "0.1.0",
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

            # Track successful reques
            await monitor.track_request(
                model=request.model,
                tokens=response.usage.get("total_tokens", 0) if response.usage else 0,
                cost=0.0,  # Implement cost calculation based on model
                response_time=time.time() - start_time,
                success=True
            )

            return response

        except Exception as e:
            # Track failed reques
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
                responses[model] = ModelResponse(
                    content=f"Error: Model {model} is not available",
                    model=model
                )
                continue

            try:
                handler = get_model_handler(model)
                response = await handler.generate(
                    request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                responses[model] = response
            except Exception as e:
                logger.error(f"Error with {model}: {str(e)}")
                responses[model] = ModelResponse(
                    content=f"Error: {str(e)}",
                    model=model
                )

        return CompareResponse(responses=responses)

    except Exception as e:
        logger.error(f"Error in compare endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics(model: Optional[str] = None):
    """Get metrics and health status for models"""
    try:
        metrics = await monitor.get_metrics(model)
        return MetricsResponse(metrics=metrics, health=monitor.health)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """Create a new chat session"""
    try:
        session = chat_manager.create_session(
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
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/sessions", response_model=List[SessionResponse])
async def list_sessions():
    """List all active chat sessions"""
    try:
        return chat_manager.list_sessions()
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session"""
    try:
        session = chat_manager.get_session(session_id)
        if not session:
            session = chat_manager.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/sessions/{session_id}/messages")
async def add_message(
    session_id: str,
    message: ChatMessage,
    background_tasks: BackgroundTasks
):
    """Add a message to a chat session"""
    try:
        session = chat_manager.get_session(session_id)
        if not session:
            session = chat_manager.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add message to session
        if message.model is None:
            model = "default_model"
        else:
            model = message.model

        session.add_message(
            role=message.role,
            content=message.content,
            model=model,
            metadata=message.metadata
        )

        # Save session in background
        background_tasks.add_task(chat_manager.save_session, session_id)

        return {"status": "success", "message_count": len(session.messages)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        if chat_manager.delete_session(session_id):
            return {"status": "success"}
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/health/check")
async def check_health(model: Optional[str] = None):
    """Check health of models"""
    try:
        if model:
            handler = get_model_handler(model)
            health = await monitor.check_health(model, handler)
            return {model: health}

        # Check all configured models
        health_status = {}
        for model_name in config.validate(value={}).keys():
            if config.validate(value={})[model_name]:
                handler = get_model_handler(model_name)
                health = await monitor.check_health(model_name, handler)
                health_status[model_name] = health
        return health_status
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MultiMindAPI:
    """API Gateway for MultiMind"""
    def __init__(self):
        self.app = FastAPI()
        self.configure_routes()

    def configure_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        # Add more routes as needed

# Export the MultiMindAPI instance
api = MultiMindAPI()

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)