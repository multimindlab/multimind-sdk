"""
Embedding model implementations for RAG system.
"""

from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Coroutine
from dataclasses import dataclass
import numpy as np
import asyncio
from ..models.base import BaseLLM

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "text-embedding-ada-002"
    batch_size: int = 100
    cache_enabled: bool = True
    device: str = "cpu"
    max_length: Optional[int] = None
    normalize: bool = True
    custom_params: Dict[str, Any] = None

class EmbeddingGenerator:
    """Main embedding generator that can use different embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding generator.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config
        self.embedder = self._get_embedder()
    
    def _get_embedder(self) -> BaseLLM:
        """Get the appropriate embedder based on configuration."""
        if "openai" in self.config.model_name.lower():
            return OpenAIEmbedder(
                model=self.config.model_name,
                batch_size=self.config.batch_size,
                cache_enabled=self.config.cache_enabled,
                **(self.config.custom_params or {})
            )
        elif "sentence" in self.config.model_name.lower():
            return SentenceT5Embedder(
                model_name=self.config.model_name,
                device=self.config.device,
                batch_size=self.config.batch_size,
                **(self.config.custom_params or {})
            )
        else:
            return HuggingFaceEmbedder(
                model_name=self.config.model_name,
                device=self.config.device,
                batch_size=self.config.batch_size,
                **(self.config.custom_params or {})
            )
    
    async def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = await self.embedder.embed(texts)
        
        if self.config.normalize:
            embeddings = self._normalize_embeddings(embeddings)
        
        return embeddings
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate([text])
        return embeddings[0]
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors."""
        normalized = []
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized.append((np.array(embedding) / norm).tolist())
            else:
                normalized.append(embedding)
        return normalized
    
    async def initialize(self) -> None:
        """Initialize the embedding generator."""
        # Any initialization logic can go here
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generator statistics."""
        return {
            "config": self.config.__dict__,
            "embedder_type": self.embedder.__class__.__name__
        }

class OpenAIEmbedder(BaseLLM):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        batch_size: int = 100,
        cache_enabled: bool = True,
        **kwargs
    ):
        """Initialize OpenAI embedder.

        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed in one batch
            cache_enabled: Whether to enable caching of embeddings
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
        self.cache_enabled = cache_enabled
        self.client = openai.AsyncOpenAI()
        self.kwargs = kwargs
        self.cache = {} if cache_enabled else None

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

    def embeddings(self, texts: List[str], reduce_dimensionality: bool = False) -> List[List[float]]:
        """Generate embeddings with optional caching and dimensionality reduction."""
        if self.cache_enabled:
            uncached_texts = [text for text in texts if text not in self.cache]
            uncached_embeddings = self._generate_embeddings(uncached_texts)
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding
            embeddings = [self.cache[text] for text in texts]
        else:
            embeddings = self._generate_embeddings(texts)

        if reduce_dimensionality:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)  # Example: Reduce to 50 dimensions
            embeddings = pca.fit_transform(embeddings).tolist()

        return embeddings

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Actual embedding generation logic."""
        # Implement embedding generation logic here
        pass

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text from the model."""
        return "Generated text"  # Placeholder implementation

    async def generate_stream(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate text stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Generated text stream"  # Placeholder implementation
        return wrapper()

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate chat completion from the model."""
        return "Chat response"  # Placeholder implementation

    async def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate chat completion stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Chat response stream"  # Placeholder implementation
        return wrapper()

    async def embeddings(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        return [[0.0]]  # Placeholder implementation

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

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text from the model."""
        return "Generated text"  # Placeholder implementation

    async def generate_stream(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate text stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Generated text stream"  # Placeholder implementation
        return wrapper()

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate chat completion from the model."""
        return "Chat response"  # Placeholder implementation

    async def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate chat completion stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Chat response stream"  # Placeholder implementation
        return wrapper()

    async def embeddings(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        return [[0.0]]  # Placeholder implementation

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

            # Convert to lis
            all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings

    async def get_quality(self) -> Optional[float]:
        """Get the quality score for this model."""
        return None  # Placeholder implementation

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text from the model."""
        return "Generated text"  # Placeholder implementation

    async def generate_stream(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate text stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Generated text stream"  # Placeholder implementation
        return wrapper()

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate chat completion from the model."""
        return "Chat response"  # Placeholder implementation

    async def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None, **kwargs) -> Coroutine[Any, Any, AsyncGenerator[str, None]]:
        """Generate chat completion stream from the model."""
        async def wrapper() -> AsyncGenerator[str, None]:
            yield "Chat response stream"  # Placeholder implementation
        return wrapper()

    async def embeddings(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the input text."""
        return [[0.0]]  # Placeholder implementation

from PIL import Image
# Optional transformers import for image embedding features
try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Image embedding features will be disabled.")

class ImageEmbedder(BaseLLM):
    """Image embedding model implementation."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize Image embedder.

        Args:
            model_name: Name of the pre-trained image embedding model.
        """
        self.model_name = model_name
        if TRANSFORMERS_AVAILABLE:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            self.model = None
            self.processor = None

    def embed(self, images: List[Image.Image]) -> List[List[float]]:
        """Generate embeddings for a list of images.

        Args:
            images: List of PIL Image objects to embed.

        Returns:
            List of embedding vectors.
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None or self.processor is None:
            raise ImportError("Transformers is required for ImageEmbedder. Please install transformers.")
        
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy().tolist()

    def process_images(self, images: List[Any]) -> Any:
        """Process images using the processor."""
        if not callable(self.processor):
            raise TypeError("Processor is not callable")
        return self.processor(images=images, return_tensors="pt", padding=True)

    def get_image_features(self, inputs: Any) -> Any:
        """Get image features from the model."""
        if not hasattr(self.model, 'get_image_features'):
            raise AttributeError("Model does not have `get_image_features` method")
        return self.model.get_image_features(**inputs)

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