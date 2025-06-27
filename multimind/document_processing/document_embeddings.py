"""
All embedding strategy classes and utilities for document embeddings.
"""
from typing import Callable, List, Any, Optional

class EmbeddingStrategy:
    """
    Generic embedding strategy wrapper.
    Args:
        embed_fn: Callable that takes a list of strings and returns a list of embeddings.
        api_key: Optional API key for embedding service (can be set at runtime).
    Example usage:
        # OpenAI
        import openai
        def openai_embed(texts, api_key):
            openai.api_key = api_key
            return [openai.Embedding.create(input=t, model="text-embedding-ada-002")['data'][0]['embedding'] for t in texts]
        emb = EmbeddingStrategy(lambda texts: openai_embed(texts, api_key="YOUR_KEY"))
        emb.embed(["hello", "world"])

        # HuggingFace
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        def hf_embed(texts, model, tokenizer):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).tolist()
        emb = EmbeddingStrategy(lambda texts: hf_embed(texts, model, tokenizer))
        emb.embed(["hello", "world"])
    """
    def __init__(self, embed_fn: Callable[[List[str]], List[Any]], api_key: Optional[str] = None):
        self.embed_fn = embed_fn
        self.api_key = api_key

    def embed(self, texts: List[str]) -> List[Any]:
        if self.api_key:
            return self.embed_fn(texts, self.api_key)
        return self.embed_fn(texts)


class ImageEmbeddingStrategy:
    """
    Embedding strategy for images. User provides an embedding function (e.g., CLIP, BLIP, custom model).
    Args:
        embed_fn: Callable that takes a list of image file paths or PIL images and returns a list of embeddings.
        model: Optional model instance (can be set at runtime).
    Example usage:
        # Using CLIP from transformers
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        def clip_embed(images, model, processor):
            inputs = processor(images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            return outputs.cpu().numpy().tolist()
        emb = ImageEmbeddingStrategy(lambda imgs: clip_embed(imgs, model, processor))
        emb.embed([Image.open('img1.png'), Image.open('img2.jpg')])
    """
    def __init__(self, embed_fn: Callable[[List[Any]], List[Any]], model: Optional[Any] = None):
        self.embed_fn = embed_fn
        self.model = model
    def embed(self, images: List[Any]) -> List[Any]:
        if self.model:
            return self.embed_fn(images, self.model)
        return self.embed_fn(images)

class AudioEmbeddingStrategy:
    """
    Embedding strategy for audio. User provides an embedding function (e.g., OpenAI Whisper, custom model).
    Args:
        embed_fn: Callable that takes a list of audio file paths or audio arrays and returns a list of embeddings.
        model: Optional model instance (can be set at runtime).
    Example usage:
        # Using a custom audio embedding model
        def audio_embed(audio_files, model):
            # Implement your audio embedding logic here
            return [model.embed(audio) for audio in audio_files]
        emb = AudioEmbeddingStrategy(lambda audios: audio_embed(audios, model))
        emb.embed(['audio1.wav', 'audio2.mp3'])
    """
    def __init__(self, embed_fn: Callable[[List[Any]], List[Any]], model: Optional[Any] = None):
        self.embed_fn = embed_fn
        self.model = model
    def embed(self, audios: List[Any]) -> List[Any]:
        if self.model:
            return self.embed_fn(audios, self.model)
        return self.embed_fn(audios)

class VideoEmbeddingStrategy:
    """
    Embedding strategy for videos. User provides an embedding function (e.g., CLIP, custom video model).
    Args:
        embed_fn: Callable that takes a list of video file paths or video arrays and returns a list of embeddings.
        model: Optional model instance (can be set at runtime).
    Example usage:
        # Using a custom video embedding model
        def video_embed(video_files, model):
            # Implement your video embedding logic here
            return [model.embed(video) for video in video_files]
        emb = VideoEmbeddingStrategy(lambda videos: video_embed(videos, model))
        emb.embed(['video1.mp4', 'video2.mov'])
    """
    def __init__(self, embed_fn: Callable[[List[Any]], List[Any]], model: Optional[Any] = None):
        self.embed_fn = embed_fn
        self.model = model
    def embed(self, videos: List[Any]) -> List[Any]:
        if self.model:
            return self.embed_fn(videos, self.model)
        return self.embed_fn(videos)

class BatchingEmbeddingStrategy:
    """
    Wrapper for batching embeddings. Splits input into batches and calls the underlying embedding strategy.
    Args:
        embedding_strategy: An embedding strategy instance (text, image, audio, video, etc.)
        batch_size: Number of items per batch
    """
    def __init__(self, embedding_strategy: Any, batch_size: int = 32):
        self.embedding_strategy = embedding_strategy
        self.batch_size = batch_size
    def embed(self, items: List[Any]) -> List[Any]:
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            results.extend(self.embedding_strategy.embed(batch))
        return results

class CachingEmbeddingStrategy:
    """
    Wrapper for caching embeddings. Caches results for repeated items.
    Args:
        embedding_strategy: An embedding strategy instance (text, image, audio, video, etc.)
    """
    def __init__(self, embedding_strategy: Any):
        self.embedding_strategy = embedding_strategy
        self.cache = {}
    def embed(self, items: List[Any]) -> List[Any]:
        uncached = [item for item in items if item not in self.cache]
        if uncached:
            new_embeds = self.embedding_strategy.embed(uncached)
            for item, emb in zip(uncached, new_embeds):
                self.cache[item] = emb
        return [self.cache[item] for item in items] 