"""
Client library for the MultiMind RAG API.
"""

from typing import List, Dict, Any, Optional, Union
import aiohttp
import json
from pathlib import Path
from datetime import datetime
import asyncio
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

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

class RAGClient:
    """Client for interacting with the MultiMind RAG API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        token: Optional[str] = None
    ):
        """Initialize the RAG client.
        
        Args:
            base_url: Base URL of the RAG API
            api_key: API key for authentication
            token: JWT token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
        elif token:
            self.headers["Authorization"] = f"Bearer {token}"
            
    async def login(self, username: str, password: str) -> str:
        """Login and get access token.
        
        Args:
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            Access token
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/token",
                data={"username": username, "password": password}
            ) as response:
                if response.status != 200:
                    raise Exception(f"Login failed: {await response.text()}")
                data = await response.json()
                self.headers["Authorization"] = f"Bearer {data['access_token']}"
                return data["access_token"]
    
    async def add_documents(
        self,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Response from the API
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/documents",
                json={"documents": [doc.dict() for doc in documents]},
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to add documents: {await response.text()}")
                return await response.json()
    
    async def add_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a file to the RAG system.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata for the file
            
        Returns:
            Response from the API
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                file_path.open("rb"),
                filename=file_path.name
            )
            if metadata:
                data.add_field("metadata", json.dumps(metadata))
                
            async with session.post(
                f"{self.base_url}/files",
                data=data,
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to add file: {await response.text()}")
                return await response.json()
    
    async def query(
        self,
        query: str,
        top_k: Optional[int] = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Query results
        """
        request = QueryRequest(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/query",
                json=request.dict(),
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Query failed: {await response.text()}")
                return await response.json()
    
    async def generate(
        self,
        query: str,
        top_k: Optional[int] = 3,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response using the RAG system.
        
        Args:
            query: Query string
            top_k: Number of documents to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            filter_metadata: Optional metadata filter
            
        Returns:
            Generated response
        """
        request = GenerateRequest(
            query=query,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            filter_metadata=filter_metadata
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=request.dict(),
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Generation failed: {await response.text()}")
                return await response.json()
    
    async def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the RAG system.
        
        Returns:
            Response from the API
        """
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/documents",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to clear documents: {await response.text()}")
                return await response.json()
    
    async def get_document_count(self) -> int:
        """Get the number of documents in the RAG system.
        
        Returns:
            Number of documents
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/documents/count",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get document count: {await response.text()}")
                data = await response.json()
                return data["count"]
    
    async def switch_model(
        self,
        model_type: str,
        model_name: str
    ) -> Dict[str, Any]:
        """Switch the model used by the RAG system.
        
        Args:
            model_type: Type of model ("openai" or "anthropic")
            model_name: Name of the model
            
        Returns:
            Response from the API
        """
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("model_type", model_type)
            data.add_field("model_name", model_name)
            
            async with session.post(
                f"{self.base_url}/models/switch",
                data=data,
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to switch model: {await response.text()}")
                return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the RAG system.
        
        Returns:
            Health status
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/health",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Health check failed: {await response.text()}")
                return await response.json()

# Example usage
async def example():
    # Initialize client
    client = RAGClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"  # or use token
    )
    
    # Login (if using JWT)
    # token = await client.login("username", "password")
    
    # Add documents
    docs = [
        Document(
            text="The RAG system provides powerful document processing.",
            metadata={"type": "introduction"}
        )
    ]
    await client.add_documents(docs)
    
    # Query
    results = await client.query("What is the RAG system?")
    print("Query results:", results)
    
    # Generate
    response = await client.generate(
        "Explain the RAG system",
        temperature=0.7
    )
    print("Generated response:", response)
    
    # Get document count
    count = await client.get_document_count()
    print("Document count:", count)
    
    # Health check
    health = await client.health_check()
    print("Health status:", health)

if __name__ == "__main__":
    asyncio.run(example()) 