"""
RAG API Server for MultiMind SDK.

This module provides RESTful API endpoints for the RAG system.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Require passlib for secure password hashing.
try:
    from passlib.context import CryptContext

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    HAS_PASSLIB = True
except ImportError:
    HAS_PASSLIB = False

from .. import __version__
from ..document_processing.base import Document
from ..embeddings.embedding import EmbeddingConfig
from ..models import ClaudeModel, OpenAIModel
from ..models.base import BaseLLM
from ..rag import RAG, RAGConfig
from ..vector_store import VectorStoreConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind RAG API",
    description=(
        "RESTful API for the MultiMind RAG system: document ingestion, retrieval, "
        "and grounded generation. The RAG backend is initialized lazily on first "
        "use; provider API keys (OPENAI_API_KEY / ANTHROPIC_API_KEY) are read from "
        "the environment at request time. Requests fail with 503 when the backend "
        "cannot be initialized. Authentication uses `X-API-Key` (when `API_KEYS` is "
        "set) or a JWT bearer token (when `JWT_SECRET` is set)."
    ),
    version=__version__,
    openapi_tags=[
        {"name": "auth", "description": "JWT token issuance"},
        {"name": "documents", "description": "Document ingestion and management"},
        {"name": "rag", "description": "Retrieval and grounded generation"},
        {"name": "models", "description": "Model management"},
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


class ErrorResponse(BaseModel):
    detail: str


ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing or invalid credentials"},
    422: {"description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
    503: {"model": ErrorResponse, "description": "RAG backend not available"},
}

# Security setup
security = HTTPBearer(auto_error=False)


# Password hashing helper
def hash_password(password: str) -> str:
    """Hash a password."""
    if not HAS_PASSLIB:
        raise RuntimeError("passlib[bcrypt] is required for secure password hashing")
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password."""
    if not HAS_PASSLIB:
        raise RuntimeError("passlib[bcrypt] is required for secure password verification")
    return pwd_context.verify(password, hashed)


JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30


def _get_api_keys() -> List[str]:
    # Read at request time so the app can start without any keys configured.
    raw = os.getenv("API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_jwt_secret() -> Optional[str]:
    return os.getenv("JWT_SECRET")


def _load_jwt_users() -> Dict[str, str]:
    """
    Load JWT users from environment variable JWT_USERS_JSON.
    Format: {"username":"hashed_password", ...}
    """
    raw_users = os.getenv("JWT_USERS_JSON")
    if not raw_users:
        return {}
    try:
        users = json.loads(raw_users)
        if isinstance(users, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in users.items()
        ):
            return users
        logger.error("JWT_USERS_JSON must be a JSON object of username->hashed_password")
        return {}
    except json.JSONDecodeError:
        logger.error("JWT_USERS_JSON is not valid JSON")
        return {}


# Global RAG instance and model
rag_instance: Optional[RAG] = None
current_model: Optional[BaseLLM] = None


# Pydantic models
class DocumentRequest(BaseModel):
    """Request model for a single document."""

    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentsRequest(BaseModel):
    """Request model for adding multiple documents."""

    documents: List[DocumentRequest] = Field(..., description="List of documents to add")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"documents": [{"text": "MultiMind is an AI SDK.", "metadata": {"source": "docs"}}]}
            ]
        }
    }


class DocumentResponse(BaseModel):
    """Response model for a document."""

    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QueryRequest(BaseModel):
    """Request model for querying."""

    query: str = Field(..., description="Query string")
    top_k: Optional[int] = Field(default=3, description="Number of results to return")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")

    model_config = {
        "json_schema_extra": {"examples": [{"query": "What is MultiMind?", "top_k": 3}]}
    }


class GenerateRequest(BaseModel):
    """Request model for generation."""

    query: str = Field(..., description="Query string")
    top_k: Optional[int] = Field(default=3, description="Number of documents to use")
    temperature: Optional[float] = Field(default=0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "Summarize the indexed documents.", "top_k": 3}]
        }
    }


class QueryResponse(BaseModel):
    """Response model for query results."""

    documents: List[DocumentResponse]
    total: int


class GenerateResponse(BaseModel):
    """Response model for generation."""

    text: str
    documents: List[DocumentResponse]


class TokenResponse(BaseModel):
    """Response model for token."""

    access_token: str
    token_type: str = "bearer"


# Authentication functions
def _auth_configured() -> bool:
    # Anonymous access is only acceptable when no auth mechanism is configured at all
    return bool(_get_api_keys()) or bool(_get_jwt_secret())


def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """Verify API key."""
    api_keys = _get_api_keys()
    if not _auth_configured():
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Verify JWT token."""
    jwt_secret = _get_jwt_secret()
    if not jwt_secret:
        raise HTTPException(status_code=503, detail="JWT authentication is not configured")
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        token = credentials.credentials
        payload = jwt.decode(token, jwt_secret, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def authenticate(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    """Authenticate using either API key or JWT token."""
    api_keys = _get_api_keys()
    # Try API key first
    if api_key and api_key in api_keys:
        return True

    # Try JWT token
    if credentials:
        try:
            verify_token(credentials)
            return True
        except HTTPException:
            pass

    if not _auth_configured():
        return True

    raise HTTPException(status_code=401, detail="Authentication required")


# Initialize RAG system
async def initialize_rag():
    """Initialize the RAG system."""
    global rag_instance, current_model

    if rag_instance is not None:
        return

    try:
        # Determine which models to use
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        logger.info("Checking for API keys")
        logger.info("OPENAI_API_KEY: %s", "Found" if openai_key else "Not found")
        logger.info("ANTHROPIC_API_KEY: %s", "Found" if anthropic_key else "Not found")

        # Default to OpenAI if available
        if openai_key:
            embedding_model_type = "openai"
            embedding_model_name = "text-embedding-ada-002"
            embedding_dimension = 1536
            embedding_api_key = openai_key
            current_model = OpenAIModel(model_name="gpt-3.5-turbo", temperature=0.7)

        elif anthropic_key:
            embedding_model_type = "openai"  # Use OpenAI for embeddings
            embedding_model_name = "text-embedding-ada-002"
            embedding_dimension = 1536
            embedding_api_key = None  # Will need OpenAI key for embeddings
            current_model = ClaudeModel(model_name="claude-3-sonnet-20240229", temperature=0.7)

        else:
            # Fallback to HuggingFace if available
            try:
                from ..models import HuggingFaceModel

                embedding_model_type = "huggingface"
                embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                embedding_dimension = 384
                embedding_api_key = None
                current_model = HuggingFaceModel(model_name="gpt2", api_key=None)
            except ImportError:
                raise ValueError(
                    "No model API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                )

        # Create vector store config
        vector_store_config = VectorStoreConfig.create_faiss_config(
            dimension=embedding_dimension, metric="cosine", index_type="flat"
        )

        # Create embedding config
        embedding_config = EmbeddingConfig(
            model_name=embedding_model_name,
            model_type=embedding_model_type,
            batch_size=32,
            max_length=512,
            normalize=True,
            device="cpu",
            cache_dir=None,
            custom_params={"api_key": embedding_api_key} if embedding_api_key else {},
        )

        # Create RAG configuration
        config = RAGConfig(
            vector_store_config=vector_store_config,
            retrieval_config={"top_k": 3, "similarity_threshold": 0.5},
            embedding_config=embedding_config,
            document_config={"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200},
        )

        # Initialize RAG system
        rag_instance = RAG(config)
        await rag_instance.initialize()

        provider = (
            "OpenAI" if openai_key else "Anthropic" if anthropic_key else "HuggingFace (Local)"
        )
        text_model = current_model.model_name if hasattr(current_model, "model_name") else "N/A"
        logger.info("RAG system initialized successfully")
        logger.info("Provider: %s", provider)
        logger.info("Text model: %s", text_model)
        logger.info("Embedding model: %s", embedding_model_name)
        logger.info("Vector store: FAISS")

        logger.info("RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


async def _ensure_rag() -> RAG:
    """Lazily initialize the RAG backend, mapping failures to a clear 503."""
    if rag_instance is not None:
        return rag_instance
    try:
        await initialize_rag()
    except Exception as e:
        logger.error(f"RAG backend unavailable: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG backend is not available. Configure OPENAI_API_KEY or "
                "ANTHROPIC_API_KEY (or install the local HuggingFace extras) and retry."
            ),
        )
    return rag_instance


# Authentication endpoints
@app.post("/token", response_model=TokenResponse, tags=["auth"], responses=ERROR_RESPONSES)
async def login(username: str = Form(...), password: str = Form(...)):
    """Get JWT token for authentication."""
    jwt_secret = _get_jwt_secret()
    jwt_users = _load_jwt_users()
    if not jwt_secret:
        raise HTTPException(status_code=503, detail="JWT authentication is not configured")
    if not jwt_users:
        raise HTTPException(status_code=503, detail="No JWT users configured")

    if username not in jwt_users:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not verify_password(password, jwt_users[username]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Create token
    expiration = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    payload = {"sub": username, "exp": expiration, "scopes": ["rag:read", "rag:write"]}

    token = jwt.encode(payload, jwt_secret, algorithm=JWT_ALGORITHM)
    return TokenResponse(access_token=token)


# Document management endpoints
@app.post(
    "/documents", response_model=Dict[str, Any], tags=["documents"], responses=ERROR_RESPONSES
)
async def add_documents(request: DocumentsRequest, authenticated: bool = Depends(authenticate)):
    """Add documents to the RAG system."""
    rag = await _ensure_rag()
    try:
        # Convert to Document objects
        documents = [
            Document(
                id=f"doc_{i}_{datetime.now().timestamp()}",
                content=doc.text,
                metadata=doc.metadata,
                source="api",
            )
            for i, doc in enumerate(request.documents)
        ]

        # Add documents
        await rag.add_documents(documents, process=True)

        logger.info("Successfully added %d document(s)", len(request.documents))

        return {
            "documents": [
                {"text": doc.text, "metadata": doc.metadata} for doc in request.documents
            ],
            "total": len(request.documents),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to add documents")


@app.post("/files", response_model=Dict[str, Any], tags=["documents"], responses=ERROR_RESPONSES)
async def add_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    authenticated: bool = Depends(authenticate),
):
    """Add a file to the RAG system."""
    rag = await _ensure_rag()
    try:
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            try:
                file_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                file_metadata = {"source": "file"}
        else:
            file_metadata = {"source": "file", "filename": file.filename}

        # Save file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        try:
            # Load and process file
            from ..document_loader.document_loader import LocalDocumentLoader

            loader = LocalDocumentLoader()
            loaded_docs = await loader.load_file(tmp_path)

            # Convert to Document objects
            documents = []
            for i, doc_content in enumerate(loaded_docs):
                if isinstance(doc_content, str):
                    doc_text = doc_content
                elif hasattr(doc_content, "content"):
                    doc_text = doc_content.content
                else:
                    doc_text = str(doc_content)

                documents.append(
                    Document(
                        id=f"file_{file.filename}_{i}_{datetime.now().timestamp()}",
                        content=doc_text,
                        metadata={**file_metadata, "filename": file.filename},
                        source=file.filename,
                    )
                )

            # Add documents
            await rag.add_documents(documents, process=True)

            return {
                "documents": [
                    {"text": f"Added file: {file.filename}", "metadata": doc.metadata}
                    for doc in documents
                ],
                "total": len(documents),
            }
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding file: {e}")
        raise HTTPException(status_code=500, detail="Failed to add file")


@app.post("/query", response_model=QueryResponse, tags=["rag"], responses=ERROR_RESPONSES)
async def query_documents(request: QueryRequest, authenticated: bool = Depends(authenticate)):
    """Query the RAG system for relevant documents."""
    rag = await _ensure_rag()
    try:
        # Build filter criteria
        filter_criteria = None
        if request.filter_metadata:
            filter_criteria = request.filter_metadata

        # Retrieve documents
        retrieved_docs = await rag.retrieve(
            request.query, k=request.top_k, filter_criteria=filter_criteria
        )

        # Convert to response format
        documents = []
        for doc in retrieved_docs:
            score = getattr(doc, "score", None)
            if score is None and hasattr(doc, "metadata") and "score" in doc.metadata:
                score = doc.metadata["score"]

            documents.append(
                DocumentResponse(
                    text=doc.content if hasattr(doc, "content") else str(doc),
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                    score=score,
                )
            )

        return QueryResponse(documents=documents, total=len(documents))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


@app.post("/generate", response_model=GenerateResponse, tags=["rag"], responses=ERROR_RESPONSES)
async def generate_response(request: GenerateRequest, authenticated: bool = Depends(authenticate)):
    """Generate a response using the RAG system."""
    rag = await _ensure_rag()
    try:
        if current_model is None:
            raise HTTPException(status_code=503, detail="No model available for generation")

        # Retrieve relevant documents
        filter_criteria = None
        if request.filter_metadata:
            filter_criteria = request.filter_metadata

        retrieved_docs = await rag.retrieve(
            request.query, k=request.top_k, filter_criteria=filter_criteria
        )

        # Build context from retrieved documents
        context = "\n\n".join(
            [doc.content if hasattr(doc, "content") else str(doc) for doc in retrieved_docs]
        )

        # Build prompt
        prompt = f"""Context:
{context}

Question: {request.query}

Answer:"""

        # Generate response
        response_text = await current_model.generate(
            prompt, temperature=request.temperature, max_tokens=request.max_tokens
        )

        # Convert documents to response format
        documents = []
        for doc in retrieved_docs:
            score = getattr(doc, "score", None)
            if score is None and hasattr(doc, "metadata") and "score" in doc.metadata:
                score = doc.metadata["score"]

            documents.append(
                DocumentResponse(
                    text=doc.content if hasattr(doc, "content") else str(doc),
                    metadata=doc.metadata if hasattr(doc, "metadata") else {},
                    score=score,
                )
            )

        return GenerateResponse(text=response_text, documents=documents)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")


@app.delete("/documents", tags=["documents"], responses=ERROR_RESPONSES)
async def clear_documents(authenticated: bool = Depends(authenticate)):
    """Clear all documents from the RAG system."""
    rag = await _ensure_rag()
    try:
        await rag.clear()

        return {"message": "All documents cleared successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear documents")


@app.get("/documents/count", tags=["documents"], responses=ERROR_RESPONSES)
async def get_document_count(authenticated: bool = Depends(authenticate)):
    """Get the number of documents in the RAG system."""
    rag = await _ensure_rag()
    try:
        # Get count from vector store - try different methods
        count = 0
        backend = rag.vector_store._get_backend()

        # Try to get count from backend metadata
        if hasattr(backend, "metadata") and backend.metadata:
            count = len(backend.metadata)
        elif hasattr(backend, "_metadata") and backend._metadata:
            count = len(backend._metadata)
        elif hasattr(backend, "index") and hasattr(backend.index, "ntotal"):
            # FAISS has ntotal attribute
            count = backend.index.ntotal
        elif hasattr(backend, "index") and hasattr(backend.index, "__len__"):
            count = len(backend.index)

        return {"count": count}

    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        # Return 0 if we can't determine the count
        return {"count": 0}


# Model management endpoints
@app.post("/models/switch", tags=["models"], responses=ERROR_RESPONSES)
async def switch_model(
    model_type: str = Form(...),
    model_name: str = Form(...),
    authenticated: bool = Depends(authenticate),
):
    """Switch the model used by the RAG system."""
    try:
        global current_model

        if model_type.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
            current_model = OpenAIModel(model_name=model_name, temperature=0.7)
        elif model_type.lower() == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")
            current_model = ClaudeModel(model_name=model_name, temperature=0.7)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        return {"message": f"Switched to {model_type} model: {model_name}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail="Failed to switch model")


# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Liveness probe (no auth; does not initialize the RAG backend)."""
    return {
        "status": "healthy",
        "version": __version__,
        "rag_initialized": rag_instance is not None,
    }


@app.get("/ready", tags=["system"], responses={503: {"model": ErrorResponse}})
async def readiness_check():
    """Readiness probe; 503 until the RAG backend has been initialized."""
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG backend not initialized")

    count = 0
    try:
        backend = rag_instance.vector_store._get_backend()
        if hasattr(backend, "metadata") and backend.metadata:
            count = len(backend.metadata)
        elif hasattr(backend, "_metadata") and backend._metadata:
            count = len(backend._metadata)
        elif hasattr(backend, "index") and hasattr(backend.index, "ntotal"):
            count = backend.index.ntotal
        elif hasattr(backend, "index") and hasattr(backend.index, "__len__"):
            count = len(backend.index)
    except Exception as count_error:
        logger.warning(f"Could not get document count: {count_error}")
        count = 0

    return {"status": "ready", "version": __version__, "document_count": count}


def start(host: str = "0.0.0.0", port: int = 8000):
    """Start the RAG API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
