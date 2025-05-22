"""
FastAPI implementation for the RAG system.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import asyncio
import json

from multimind.rag.rag import RAG
from multimind.rag.document import Document
from multimind.rag.embeddings import get_embedder
from multimind.models.openai import OpenAIModel
from multimind.api.auth import (
    User, Token, create_access_token, get_current_active_user,
    check_scope, ACCESS_TOKEN_EXPIRE_MINUTES, timedelta
)

app = FastAPI(title="MultiMind RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_instance: Optional[RAG] = None

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class BatchDocumentRequest(BaseModel):
    documents: List[DocumentRequest]

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_metadata: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    query: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    filter_metadata: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class QueryResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

class GenerateResponse(BaseModel):
    response: str
    documents: List[DocumentResponse]

async def get_rag() -> RAG:
    """Get or create RAG instance."""
    global rag_instance
    if rag_instance is None:
        # Initialize with default settings
        model = OpenAIModel(model_name="gpt-3.5-turbo")
        embedder = get_embedder("openai")
        rag_instance = RAG(
            embedder=embedder,
            vector_store="faiss",
            model=model
        )
    return rag_instance

@app.post("/documents")
async def add_documents(
    request: BatchDocumentRequest,
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
) -> Dict[str, int]:
    """Add documents to the RAG system."""
    try:
        # Convert request to documents and metadata
        docs = [doc.text for doc in request.documents]
        metadata = [doc.metadata or {} for doc in request.documents]

        # Add documents
        await rag.add_documents(docs, metadata=metadata)
        count = len(docs)
        return {"document_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:read"))
) -> QueryResponse:
    """Query documents from the RAG system."""
    try:
        # Search documents
        results = await rag.search(request.query, k=request.top_k or 3)

        # Convert to response format
        documents = [
            DocumentResponse(
                text=doc["text"],
                metadata=doc["metadata"],
                score=doc.get("score")
            )
            for doc in results
        ]
        return QueryResponse(documents=documents, total=len(documents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(
    request: GenerateRequest,
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:read"))
) -> GenerateResponse:
    """Generate a response using the RAG system."""
    try:
        # First search for relevant documents
        results = await rag.search(request.query, k=3)
        
        # Generate response using context
        response = await rag.query(
            query=request.query,
            context=results,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Convert search results to response format
        documents = [
            DocumentResponse(
                text=doc["text"],
                metadata=doc["metadata"],
                score=doc.get("score")
            )
            for doc in results
        ]
        return GenerateResponse(response=response, documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/switch")
async def switch_model(
    model_type: str = Form(...),
    model_name: str = Form(...),
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
):
    """Switch the model used by the RAG system."""
    try:
        if model_type == "openai":
            rag.model = OpenAIModel(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return {"message": f"Switched to {model_type} model: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/clear")
async def clear_documents(
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
):
    """Clear all documents from the RAG system."""
    try:
        await rag.vector_store.clear()
        return {"message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(
    rag: RAG = Depends(get_rag)
) -> Dict[str, Union[str, int, Dict]]:
    """Check the health of the RAG system."""
    try:
        # Check vector store
        doc_count = await rag.vector_store.get_document_count()
        
        # Test embeddings generation
        embedding_test = await rag.embedder.embeddings("test")
        embedding_dim = len(embedding_test[0]) if isinstance(embedding_test[0], list) else len(embedding_test)
        
        # Test model generation (with timeout)
        model_healthy = True
        model_error = None
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout
                await rag.model.generate("test", max_tokens=10)
        except Exception as e:
            model_healthy = False
            model_error = str(e)
        
        return {
            "status": "healthy" if model_healthy else "degraded",
            "document_count": doc_count,
            "embedding_dimension": embedding_dim,
            "vector_store_type": rag.vector_store.__class__.__name__,
            "model": {
                "name": rag.model.model_name if rag.model else None,
                "status": "healthy" if model_healthy else "error",
                "error": model_error
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )

# Mock user database for testing purposes
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "password": "secret"
    }
}

# Add authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """Get access token for authentication."""
    # In production, verify against real user database
    user = fake_users_db.get(form_data.username)
    if not user or form_data.password != "secret":  # Use proper password verification
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)