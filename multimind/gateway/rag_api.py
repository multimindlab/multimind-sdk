"""
RAG API Server for MultiMind SDK.

This module provides RESTful API endpoints for the RAG system.
"""

import os
import logging
import json
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt

# Try to import passlib for password hashing, fallback to simple hash
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    HAS_PASSLIB = True
except ImportError:
    HAS_PASSLIB = False

from ..rag import RAG, RAGConfig
from ..document_processing.base import Document
from ..vector_store import VectorStoreConfig
from ..embeddings.embedding import EmbeddingConfig, EmbeddingType
from ..models import OpenAIModel, ClaudeModel
from ..models.base import BaseLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind RAG API",
    description="RESTful API for the MultiMind RAG system",
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

# Security setup
security = HTTPBearer(auto_error=False)

# Password hashing helper
def hash_password(password: str) -> str:
    """Hash a password."""
    if HAS_PASSLIB:
        return pwd_context.hash(password)
    else:
        # Simple hash fallback (not secure, but works for development)
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password."""
    if HAS_PASSLIB:
        return pwd_context.verify(password, hashed)
    else:
        # Simple hash verification fallback
        return hashlib.sha256(password.encode()).hexdigest() == hashed

# Get API keys from environment
API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30

# Default users for JWT authentication (in production, use a database)
DEFAULT_USERS = {
    "testuser": hash_password("secret"),
    "admin": hash_password("admin123")
}

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


class GenerateRequest(BaseModel):
    """Request model for generation."""
    query: str = Field(..., description="Query string")
    top_k: Optional[int] = Field(default=3, description="Number of documents to use")
    temperature: Optional[float] = Field(default=0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


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
def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """Verify API key."""
    if not API_KEYS:
        # If no API keys configured, allow access (for development)
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def authenticate(api_key: Optional[str] = Header(None, alias="X-API-Key"),
                 credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Authenticate using either API key or JWT token."""
    # Try API key first
    if api_key and api_key in API_KEYS:
        return True
    
    # Try JWT token
    if credentials:
        try:
            verify_token(credentials)
            return True
        except HTTPException:
            pass
    
    # If no API keys configured, allow access (for development)
    if not API_KEYS:
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
        
        print("\n📋 Checking for API keys...")
        print(f"   OPENAI_API_KEY: {'✅ Found' if openai_key else '❌ Not found'}")
        print(f"   ANTHROPIC_API_KEY: {'✅ Found' if anthropic_key else '❌ Not found'}")
        
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
                raise ValueError("No model API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Create vector store config
        vector_store_config = VectorStoreConfig.create_faiss_config(
            dimension=embedding_dimension,
            metric="cosine",
            index_type="flat"
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
            custom_params={"api_key": embedding_api_key} if embedding_api_key else {}
        )
        
        # Create RAG configuration
        config = RAGConfig(
            vector_store_config=vector_store_config,
            retrieval_config={"top_k": 3, "similarity_threshold": 0.5},
            embedding_config=embedding_config,
            document_config={"min_chunk_size": 100, "max_chunk_size": 1000, "chunk_overlap": 200}
        )
        
        # Initialize RAG system
        rag_instance = RAG(config)
        await rag_instance.initialize()
        
        print("\n" + "="*60)
        print("✅ RAG SYSTEM INITIALIZED SUCCESSFULLY")
        print("="*60)
        print(f"📊 Provider: {'OpenAI' if openai_key else 'Anthropic' if anthropic_key else 'HuggingFace (Local)'}")
        print(f"📝 Text Model: {current_model.model_name if hasattr(current_model, 'model_name') else 'N/A'}")
        print(f"🔤 Embedding Model: {embedding_model_name}")
        print(f"📦 Vector Store: FAISS")
        print("="*60 + "\n")
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    await initialize_rag()


# Authentication endpoints
@app.post("/token", response_model=TokenResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    """Get JWT token for authentication."""
    # In production, verify against database
    if username not in DEFAULT_USERS:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not verify_password(password, DEFAULT_USERS[username]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create token
    expiration = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    payload = {
        "sub": username,
        "exp": expiration,
        "scopes": ["rag:read", "rag:write"]
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return TokenResponse(access_token=token)


# Document management endpoints
@app.post("/documents", response_model=Dict[str, Any])
async def add_documents(request: DocumentsRequest, authenticated: bool = Depends(authenticate)):
    """Add documents to the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        # Convert to Document objects
        documents = [
            Document(
                id=f"doc_{i}_{datetime.now().timestamp()}",
                content=doc.text,
                metadata=doc.metadata,
                source="api"
            )
            for i, doc in enumerate(request.documents)
        ]
        
        # Add documents
        await rag_instance.add_documents(documents, process=True)
        
        print(f"   ✅ Successfully added {len(request.documents)} document(s)")
        
        return {
            "documents": [
                {"text": doc.text, "metadata": doc.metadata}
                for doc in request.documents
            ],
            "total": len(request.documents)
        }
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")


@app.post("/files", response_model=Dict[str, Any])
async def add_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    authenticated: bool = Depends(authenticate)
):
    """Add a file to the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
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
                elif hasattr(doc_content, 'content'):
                    doc_text = doc_content.content
                else:
                    doc_text = str(doc_content)
                
                documents.append(Document(
                    id=f"file_{file.filename}_{i}_{datetime.now().timestamp()}",
                    content=doc_text,
                    metadata={**file_metadata, "filename": file.filename},
                    source=file.filename
                ))
            
            # Add documents
            await rag_instance.add_documents(documents, process=True)
            
            return {
                "documents": [
                    {
                        "text": f"Added file: {file.filename}",
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "total": len(documents)
            }
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as e:
        logger.error(f"Error adding file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add file: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, authenticated: bool = Depends(authenticate)):
    """Query the RAG system for relevant documents."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        # Build filter criteria
        filter_criteria = None
        if request.filter_metadata:
            filter_criteria = request.filter_metadata
        
        # Retrieve documents
        retrieved_docs = await rag_instance.retrieve(
            request.query,
            k=request.top_k,
            filter_criteria=filter_criteria
        )
        
        # Convert to response format
        documents = []
        for doc in retrieved_docs:
            score = getattr(doc, "score", None)
            if score is None and hasattr(doc, "metadata") and "score" in doc.metadata:
                score = doc.metadata["score"]
            
            documents.append(DocumentResponse(
                text=doc.content if hasattr(doc, "content") else str(doc),
                metadata=doc.metadata if hasattr(doc, "metadata") else {},
                score=score
            ))
        
        return QueryResponse(documents=documents, total=len(documents))
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest, authenticated: bool = Depends(authenticate)):
    """Generate a response using the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        if current_model is None:
            raise HTTPException(status_code=500, detail="No model available for generation")
        
        # Retrieve relevant documents
        filter_criteria = None
        if request.filter_metadata:
            filter_criteria = request.filter_metadata
        
        retrieved_docs = await rag_instance.retrieve(
            request.query,
            k=request.top_k,
            filter_criteria=filter_criteria
        )
        
        # Build context from retrieved documents
        context = "\n\n".join([
            doc.content if hasattr(doc, "content") else str(doc)
            for doc in retrieved_docs
        ])
        
        # Build prompt
        prompt = f"""Context:
{context}

Question: {request.query}

Answer:"""
        
        # Generate response
        response_text = await current_model.generate(
            prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Convert documents to response format
        documents = []
        for doc in retrieved_docs:
            score = getattr(doc, "score", None)
            if score is None and hasattr(doc, "metadata") and "score" in doc.metadata:
                score = doc.metadata["score"]
            
            documents.append(DocumentResponse(
                text=doc.content if hasattr(doc, "content") else str(doc),
                metadata=doc.metadata if hasattr(doc, "metadata") else {},
                score=score
            ))
        
        return GenerateResponse(text=response_text, documents=documents)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.delete("/documents")
async def clear_documents(authenticated: bool = Depends(authenticate)):
    """Clear all documents from the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        await rag_instance.clear()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@app.get("/documents/count")
async def get_document_count(authenticated: bool = Depends(authenticate)):
    """Get the number of documents in the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        # Get count from vector store - try different methods
        count = 0
        backend = rag_instance.vector_store._get_backend()
        
        # Try to get count from backend metadata
        if hasattr(backend, 'metadata') and backend.metadata:
            count = len(backend.metadata)
        elif hasattr(backend, '_metadata') and backend._metadata:
            count = len(backend._metadata)
        elif hasattr(backend, 'index') and hasattr(backend.index, 'ntotal'):
            # FAISS has ntotal attribute
            count = backend.index.ntotal
        elif hasattr(backend, 'index') and hasattr(backend.index, '__len__'):
            count = len(backend.index)
        
        return {"count": count}
        
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        # Return 0 if we can't determine the count
        return {"count": 0}


# Model management endpoints
@app.post("/models/switch")
async def switch_model(
    model_type: str = Form(...),
    model_name: str = Form(...),
    authenticated: bool = Depends(authenticate)
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
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health of the RAG system."""
    try:
        if rag_instance is None:
            await initialize_rag()
        
        # Get document count using the same method as get_document_count endpoint
        count = 0
        try:
            backend = rag_instance.vector_store._get_backend()
            if hasattr(backend, 'metadata') and backend.metadata:
                count = len(backend.metadata)
            elif hasattr(backend, '_metadata') and backend._metadata:
                count = len(backend._metadata)
            elif hasattr(backend, 'index') and hasattr(backend.index, 'ntotal'):
                count = backend.index.ntotal
            elif hasattr(backend, 'index') and hasattr(backend.index, '__len__'):
                count = len(backend.index)
        except Exception as count_error:
            logger.warning(f"Could not get document count: {count_error}")
            count = 0
        
        return {
            "status": "healthy",
            "document_count": count
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def start(host: str = "0.0.0.0", port: int = 8000):
    """Start the RAG API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
