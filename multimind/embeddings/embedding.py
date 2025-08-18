"""
Advanced embedding module with support for multiple models and multi-vector embeddings.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import numpy as np
from datetime import datetime
try:
    import torch
except ImportError:
    torch = None
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import openai
except ImportError:
    openai = None

# Graceful import for optional dependencies
try:
    import cohere
    _HAS_COHERE = True
except ImportError:
    _HAS_COHERE = False

from ..models.base import BaseLLM

@dataclass
class Embedding:
    """Represents an embedding vector with metadata."""
    vector: List[float]
    text: str
    model_name: str
    model_type: str
    metadata: Dict[str, Any]
    created_at: datetime
    embedding_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate embedding after initialization."""
        if not isinstance(self.vector, list):
            raise ValueError("Embedding vector must be a list")
        if not all(isinstance(x, (int, float)) for x in self.vector):
            raise ValueError("Embedding vector must contain numbers")
        if not isinstance(self.text, str):
            raise ValueError("Embedding text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Embedding metadata must be a dictionary")

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    model_type: str
    batch_size: int
    max_length: int
    normalize: bool
    device: str
    cache_dir: Optional[str]
    custom_params: Dict[str, Any]

@dataclass
class MultiVectorEmbedding:
    """Multi-vector embedding for a document."""
    title_embedding: List[float]
    content_embedding: List[float]
    summary_embedding: Optional[List[float]]
    metadata_embedding: Optional[List[float]]
    combined_embedding: List[float]
    metadata: Dict[str, Any]

class EmbeddingType(Enum):
    """Types of embedding models."""
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    INSTRUCTOR = "instructor"
    CUSTOM = "custom"

class EmbeddingModel:
    """Advanced embedding system with multiple model support."""

    def __init__(
        self,
        model_type: EmbeddingType,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize embedding model.
        
        Args:
            model_type: Type of embedding model
            model_name: Name of the model
            api_key: Optional API key for cloud models
            **kwargs: Additional parameters
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        
        # Initialize model based on type
        if model_type == EmbeddingType.OPENAI:
            if not api_key:
                raise ValueError("OpenAI API key required")
            openai.api_key = api_key
            self.model = None  # OpenAI uses API calls
        
        elif model_type == EmbeddingType.COHERE:
            if not _HAS_COHERE:
                raise ImportError("Cohere package not installed. Install with: pip install cohere")
            if not api_key:
                raise ValueError("Cohere API key required")
            self.model = cohere.Client(api_key)
        
        elif model_type == EmbeddingType.HUGGINGFACE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        
        elif model_type == EmbeddingType.SENTENCE_TRANSFORMER:
            self.model = SentenceTransformer(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        
        elif model_type == EmbeddingType.INSTRUCTOR:
            self.model = AutoModel.from_pretrained(
                "hkunlp/instructor-xl",
                trust_remote_code=True
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        
        else:  # CUSTOM
            raise ValueError("Custom model initialization not implemented")

    async def generate_embedding(
        self,
        text: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            config: Optional embedding configuration
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector
        """
        if config is None:
            config = self._get_default_config()
        
        # Generate embedding based on model type
        if self.model_type == EmbeddingType.OPENAI:
            return await self._generate_openai_embedding(text, config)
        
        elif self.model_type == EmbeddingType.COHERE:
            return await self._generate_cohere_embedding(text, config)
        
        elif self.model_type == EmbeddingType.HUGGINGFACE:
            return await self._generate_huggingface_embedding(text, config)
        
        elif self.model_type == EmbeddingType.SENTENCE_TRANSFORMER:
            return await self._generate_sentence_transformer_embedding(text, config)
        
        elif self.model_type == EmbeddingType.INSTRUCTOR:
            return await self._generate_instructor_embedding(text, config)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def generate_multi_vector_embedding(
        self,
        document: Dict[str, Any],
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ) -> MultiVectorEmbedding:
        """
        Generate multi-vector embedding for document.
        
        Args:
            document: Document to embed
            config: Optional embedding configuration
            **kwargs: Additional parameters
            
        Returns:
            Multi-vector embedding
        """
        if config is None:
            config = self._get_default_config()
        
        # Generate embeddings for different parts
        title_embedding = await self.generate_embedding(
            document["title"],
            config
        )
        
        content_embedding = await self.generate_embedding(
            document["content"],
            config
        )
        
        summary_embedding = None
        if "summary" in document:
            summary_embedding = await self.generate_embedding(
                document["summary"],
                config
            )
        
        metadata_embedding = None
        if "metadata" in document:
            metadata_text = json.dumps(document["metadata"])
            metadata_embedding = await self.generate_embedding(
                metadata_text,
                config
            )
        
        # Generate combined embedding
        combined_text = f"""
        Title: {document['title']}
        Content: {document['content']}
        Summary: {document.get('summary', '')}
        Metadata: {json.dumps(document.get('metadata', {}))}
        """
        
        combined_embedding = await self.generate_embedding(
            combined_text,
            config
        )
        
        return MultiVectorEmbedding(
            title_embedding=title_embedding,
            content_embedding=content_embedding,
            summary_embedding=summary_embedding,
            metadata_embedding=metadata_embedding,
            combined_embedding=combined_embedding,
            metadata={
                "model": self.model_name,
                "timestamp": datetime.now().timestamp(),
                **kwargs
            }
        )

    async def generate_batch_embeddings(
        self,
        texts: List[str],
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            config: Optional embedding configuration
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        if config is None:
            config = self._get_default_config()
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i:i + config.batch_size]
            
            if self.model_type == EmbeddingType.OPENAI:
                batch_embeddings = await self._generate_openai_batch_embeddings(
                    batch,
                    config
                )
            
            elif self.model_type == EmbeddingType.COHERE:
                batch_embeddings = await self._generate_cohere_batch_embeddings(
                    batch,
                    config
                )
            
            elif self.model_type == EmbeddingType.HUGGINGFACE:
                batch_embeddings = await self._generate_huggingface_batch_embeddings(
                    batch,
                    config
                )
            
            elif self.model_type == EmbeddingType.SENTENCE_TRANSFORMER:
                batch_embeddings = await self._generate_sentence_transformer_batch_embeddings(
                    batch,
                    config
                )
            
            elif self.model_type == EmbeddingType.INSTRUCTOR:
                batch_embeddings = await self._generate_instructor_batch_embeddings(
                    batch,
                    config
                )
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def _get_default_config(self) -> EmbeddingConfig:
        """Get default embedding configuration."""
        return EmbeddingConfig(
            model_name=self.model_name,
            model_type=self.model_type.value,
            batch_size=32,
            max_length=512,
            normalize=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir=None,
            custom_params={}
        )

    async def _generate_openai_embedding(
        self,
        text: str,
        config: EmbeddingConfig
    ) -> List[float]:
        """Generate embedding using OpenAI."""
        response = await openai.Embedding.acreate(
            input=text,
            model=self.model_name,
            **config.custom_params
        )
        
        embedding = response["data"][0]["embedding"]
        
        if config.normalize:
            embedding = self._normalize_embedding(embedding)
        
        return embedding

    async def _generate_cohere_embedding(
        self,
        text: str,
        config: EmbeddingConfig
    ) -> List[float]:
        """Generate embedding using Cohere."""
        response = self.model.embed(
            texts=[text],
            model=self.model_name,
            **config.custom_params
        )
        
        embedding = response.embeddings[0]
        
        if config.normalize:
            embedding = self._normalize_embedding(embedding)
        
        return embedding

    async def _generate_huggingface_embedding(
        self,
        text: str,
        config: EmbeddingConfig
    ) -> List[float]:
        """Generate embedding using HuggingFace model."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        
        if config.normalize:
            embedding = self._normalize_embedding(embedding)
        
        return embedding.tolist()

    async def _generate_sentence_transformer_embedding(
        self,
        text: str,
        config: EmbeddingConfig
    ) -> List[float]:
        """Generate embedding using SentenceTransformer."""
        embedding = self.model.encode(
            text,
            max_length=config.max_length,
            normalize_embeddings=config.normalize,
            **config.custom_params
        )
        
        return embedding.tolist()

    async def _generate_instructor_embedding(
        self,
        text: str,
        config: EmbeddingConfig
    ) -> List[float]:
        """Generate embedding using Instructor model."""
        # Format instruction
        instruction = "Represent the following text for retrieval:"
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode(
                [[instruction, text]],
                max_length=config.max_length,
                normalize_embeddings=config.normalize,
                **config.custom_params
            )[0]
        
        return embedding.tolist()

    async def _generate_openai_batch_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> List[List[float]]:
        """Generate batch embeddings using OpenAI."""
        response = await openai.Embedding.acreate(
            input=texts,
            model=self.model_name,
            **config.custom_params
        )
        
        embeddings = [item["embedding"] for item in response["data"]]
        
        if config.normalize:
            embeddings = [
                self._normalize_embedding(embedding)
                for embedding in embeddings
            ]
        
        return embeddings

    async def _generate_cohere_batch_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> List[List[float]]:
        """Generate batch embeddings using Cohere."""
        response = self.model.embed(
            texts=texts,
            model=self.model_name,
            **config.custom_params
        )
        
        embeddings = response.embeddings
        
        if config.normalize:
            embeddings = [
                self._normalize_embedding(embedding)
                for embedding in embeddings
            ]
        
        return embeddings

    async def _generate_huggingface_batch_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> List[List[float]]:
        """Generate batch embeddings using HuggingFace model."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        if config.normalize:
            embeddings = [
                self._normalize_embedding(embedding)
                for embedding in embeddings
            ]
        
        return embeddings.tolist()

    async def _generate_sentence_transformer_batch_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> List[List[float]]:
        """Generate batch embeddings using SentenceTransformer."""
        embeddings = self.model.encode(
            texts,
            max_length=config.max_length,
            normalize_embeddings=config.normalize,
            **config.custom_params
        )
        
        return embeddings.tolist()

    async def _generate_instructor_batch_embeddings(
        self,
        texts: List[str],
        config: EmbeddingConfig
    ) -> List[List[float]]:
        """Generate batch embeddings using Instructor model."""
        # Format instructions
        instruction = "Represent the following text for retrieval:"
        inputs = [[instruction, text] for text in texts]
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model.encode(
                inputs,
                max_length=config.max_length,
                normalize_embeddings=config.normalize,
                **config.custom_params
            )
        
        return embeddings.tolist()

    def _normalize_embedding(
        self,
        embedding: Union[List[float], np.ndarray]
    ) -> List[float]:
        """Normalize embedding vector."""
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding.tolist()
        
        normalized = embedding / norm
        return normalized.tolist()

# Utility functions for semantic voting
async def get_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """Get embedding for text using default model."""
    try:
        model = SentenceTransformer(model_name)
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        # Fallback to simple embedding
        return [0.1] * 384  # Default dimension

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        return float(similarity)
    except Exception:
        return 0.0 