"""
Embedding model implementations for RAG system.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..models.base import BaseLLM

class OpenAIEmbedder(BaseLLM):
    """OpenAI embedding model implementation."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 100,
        **kwargs
    ):
        """Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed in one batch
            **kwargs: Additional arguments for OpenAI API
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )
            
        self.model = model
        self.batch_size = batch_size
        self.client = openai.AsyncOpenAI()
        self.kwargs = kwargs
        
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for embedding API
            
        Returns:
            List of embedding vectors
        """
        # Combine kwargs
        api_kwargs = {**self.kwargs, **kwargs}
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Call OpenAI API
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                **api_kwargs
            )
            
            # Extract embeddings
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

class HuggingFaceEmbedder(BaseLLM):
    """HuggingFace embedding model implementation."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize HuggingFace embedder.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Number of texts to embed in one batch
            **kwargs: Additional arguments for model
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError(
                "Transformers and PyTorch are required. "
                "Install with: pip install transformers torch"
            )
            
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)
        self.model.eval()
        
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for model
            
        Returns:
            List of embedding vectors
        """
        import torch
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                **kwargs
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding or mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            # Convert to list and move to CPU
            batch_embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

class SentenceT5Embedder(BaseLLM):
    """Sentence-T5 embedding model implementation."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/sentence-t5-base",
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs
    ):
        """Initialize Sentence-T5 embedder.
        
        Args:
            model_name: Sentence-T5 model name
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Number of texts to embed in one batch
            **kwargs: Additional arguments for model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Sentence-Transformers is required. "
                "Install with: pip install sentence-transformers"
            )
            
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for model
            
        Returns:
            List of embedding vectors
        """
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                **kwargs
            )
            
            # Convert to list
            all_embeddings.extend(batch_embeddings.tolist())
            
        return all_embeddings

def get_embedder(
    embedder_type: str,
    **kwargs
) -> BaseLLM:
    """Factory function to create embedder instances.
    
    Args:
        embedder_type: Type of embedder ('openai', 'huggingface', or 'sentence-t5')
        **kwargs: Arguments for embedder initialization
        
    Returns:
        Initialized embedder instance
        
    Raises:
        ValueError: If embedder_type is not supported
    """
    embedders = {
        "openai": OpenAIEmbedder,
        "huggingface": HuggingFaceEmbedder,
        "sentence-t5": SentenceT5Embedder
    }
    
    if embedder_type not in embedders:
        raise ValueError(
            f"Unsupported embedder type: {embedder_type}. "
            f"Supported types: {list(embedders.keys())}"
        )
        
    return embedders[embedder_type](**kwargs) 