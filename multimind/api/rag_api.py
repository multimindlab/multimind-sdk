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

from multimind.rag import RAG, Document
from multimind.rag.embeddings import get_embedder
from multimind.models import OpenAIModel, AnthropicModel
from multimind.api.auth import (
    User, Token, create_access_token, get_current_active_user,
    check_scope, ACCESS_TOKEN_EXPIRE_MINUTES, timedelta
)

# Initialize FastAPI app
app = FastAPI(
    title="MultiMind RAG API",
    description="API for the MultiMind SDK's RAG system",
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

# Pydantic models for request/response
class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_metadata: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
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
    text: str
    documents: List[DocumentResponse]

# Global RAG instance
rag_instance: Optional[RAG] = None

async def get_rag() -> RAG:
    """Get or create RAG instance."""
    global rag_instance
    if rag_instance is None:
        # Initialize with default settings
        model = OpenAIModel(model="gpt-3.5-turbo")
        embedder = get_embedder("openai")
        rag_instance = RAG(
            embedder=embedder,
            vector_store="faiss",
            model=model
        )
    return rag_instance

# API endpoints
@app.post("/documents", response_model=QueryResponse)
async def add_documents(
    documents: List[DocumentRequest],
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
):
    """Add documents to the RAG system."""
    try:
        docs = [
            Document(text=doc.text, metadata=doc.metadata)
            for doc in documents
        ]
        await rag.add_documents(docs)
        return QueryResponse(
            documents=[
                DocumentResponse(
                    text=doc.text,
                    metadata=doc.metadata
                )
                for doc in docs
            ],
            total=len(docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files", response_model=QueryResponse)
async def add_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
):
    """Add a file to the RAG system."""
    try:
        # Parse metadata if provided
        file_metadata = json.loads(metadata) if metadata else {}
        
        # Save file temporarily
        file_path = Path(f"temp_{file.filename}")
        try:
            content = await file.read()
            file_path.write_bytes(content)
            
            # Add file to RAG
            await rag.add_file(file_path, metadata=file_metadata)
            
            # Get document count
            count = await rag.get_document_count()
            
            return QueryResponse(
                documents=[
                    DocumentResponse(
                        text=f"Added file: {file.filename}",
                        metadata=file_metadata
                    )
                ],
                total=count
            )
        finally:
            # Clean up temporary file
            if file_path.exists():
                file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:read"))
):
    """Query the RAG system."""
    try:
        results = await rag.query(
            request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )
        
        return QueryResponse(
            documents=[
                DocumentResponse(
                    text=doc.text,
                    metadata=doc.metadata,
                    score=score
                )
                for doc, score in results
            ],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:read"))
):
    """Generate a response using the RAG system."""
    try:
        # Get relevant documents
        results = await rag.query(
            request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )
        
        # Generate response
        response = await rag.generate(
            request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerateResponse(
            text=response,
            documents=[
                DocumentResponse(
                    text=doc.text,
                    metadata=doc.metadata,
                    score=score
                )
                for doc, score in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents(
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:write"))
):
    """Clear all documents from the RAG system."""
    try:
        await rag.clear()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/count")
async def get_document_count(
    rag: RAG = Depends(get_rag),
    current_user: User = Depends(check_scope("rag:read"))
):
    """Get the number of documents in the RAG system."""
    try:
        count = await rag.get_document_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
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
            rag.model = OpenAIModel(model=model_name)
        elif model_type == "anthropic":
            rag.model = AnthropicModel(model=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return {"message": f"Switched to {model_type} model: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check(
    current_user: User = Depends(get_current_active_user)
):
    """Check the health of the RAG system."""
    try:
        rag = await get_rag()
        count = await rag.get_document_count()
        return {
            "status": "healthy",
            "document_count": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

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