"""
Fluent RAG API for building and executing RAG pipelines.
"""

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from multimind import EmbeddingStandardizer, VectorStore

from ..core.router import Router, TaskType


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    vector_store: VectorStore
    embedding_provider: str
    embedding_model: str
    generation_provider: str
    generation_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_results: int = 5
    metadata: Dict[str, Any] = {}


class RAGResult(BaseModel):
    """Result from RAG pipeline."""

    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}


class RAGPipeline:
    """Fluent RAG pipeline builder."""

    def __init__(self, router: Router, config: RAGConfig):
        """Initialize the RAG pipeline."""
        self.router = router
        self.config = config
        self.standardizer = EmbeddingStandardizer()
        self._steps: List[Callable] = []
        self._context: Dict[str, Any] = {}

    def load_documents(
        self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> "RAGPipeline":
        """Load documents into the pipeline."""

        async def _load():
            # Chunk documents
            chunks = []
            chunk_metadata = []
            chunk_documents = []
            for i, doc in enumerate(documents):
                doc_chunks = self._chunk_text(doc)
                chunks.extend(doc_chunks)
                # Create metadata with text for each chunk
                for chunk in doc_chunks:
                    chunk_meta = {}
                    if metadata:
                        chunk_meta.update(metadata[i])
                    chunk_meta["text"] = chunk
                    chunk_metadata.append(chunk_meta)
                    # Create document dict for vector store
                    chunk_documents.append({"content": chunk})

            # Generate embeddings
            embeddings = []
            for chunk in chunks:
                result = await self.router.route(
                    TaskType.EMBEDDINGS,
                    chunk,
                    provider=self.config.embedding_provider,
                    model=self.config.embedding_model,
                )
                embeddings.append(result.embedding)

            # Standardize embeddings
            # Get dimension from vector store config
            target_dimension = self.config.vector_store.config.get("dimension")
            if target_dimension is None:
                # Fallback: use the dimension of the first embedding
                target_dimension = len(embeddings[0]) if embeddings else 1536

            standardized = [
                self.standardizer.standardize(emb, len(emb), target_dimension=target_dimension)
                for emb in embeddings
            ]

            # Add to vector store
            vector_ids = await self.config.vector_store.add_vectors(
                standardized, chunk_metadata, chunk_documents
            )

            self._context["chunks"] = chunks
            self._context["vector_ids"] = vector_ids

        self._steps.append(_load)
        return self

    def query(self, query: str, **kwargs) -> "RAGPipeline":
        """Add a query to the pipeline."""

        async def _query():
            # Generate query embedding
            result = await self.router.route(
                TaskType.EMBEDDINGS,
                query,
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
            )

            # Standardize query embedding
            # Get dimension from vector store config
            target_dimension = self.config.vector_store.config.get("dimension")
            if target_dimension is None:
                # Fallback: use the dimension of the query embedding
                target_dimension = len(result.embedding) if result.embedding else 1536

            query_embedding = self.standardizer.standardize(
                result.embedding, len(result.embedding), target_dimension=target_dimension
            )

            # Search vector store
            results = await self.config.vector_store.search(
                query_embedding, k=self.config.max_results, **kwargs
            )

            self._context["query"] = query
            self._context["search_results"] = results

        self._steps.append(_query)
        return self

    def generate(
        self,
        prompt_template: str = "Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
        **kwargs,
    ) -> "RAGPipeline":
        """Generate an answer using the context."""

        async def _generate():
            # Prepare context using get_content() for consistent content extraction
            context = "\n\n".join([r.get_content() for r in self._context["search_results"]])

            # Format prompt
            prompt = prompt_template.format(context=context, query=self._context["query"])

            # Generate answer
            result = await self.router.route(
                TaskType.TEXT_GENERATION,
                prompt,
                provider=self.config.generation_provider,
                model=self.config.generation_model,
                **kwargs,
            )

            self._context["answer"] = result.text
            self._context["sources"] = [
                {
                    "text": r.get_content(),
                    "metadata": r.metadata if hasattr(r, "metadata") else {},
                    "score": r.score if hasattr(r, "score") else 0.0,
                }
                for r in self._context["search_results"]
            ]

        self._steps.append(_generate)
        return self

    def filter(self, filter_fn: Callable[[Any], bool]) -> "RAGPipeline":
        """Filter search results."""

        async def _filter():
            self._context["search_results"] = [
                r for r in self._context["search_results"] if filter_fn(r)
            ]

        self._steps.append(_filter)
        return self

    def transform(self, transform_fn: Callable[[Any], Any]) -> "RAGPipeline":
        """Transform search results."""

        async def _transform():
            self._context["search_results"] = [
                transform_fn(r) for r in self._context["search_results"]
            ]

        self._steps.append(_transform)
        return self

    async def execute(self) -> RAGResult:
        """Execute the pipeline."""
        # Run all steps
        for step in self._steps:
            await step()

        return RAGResult(
            answer=self._context["answer"], sources=self._context["sources"], metadata=self._context
        )

    def _chunk_text(
        self, text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.config.chunk_overlap

        # Ensure overlap is smaller than chunk size to guarantee progress
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 2)

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end > text_len:
                end = text_len

            # Find the last space in the chunk
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space

            chunks.append(text[start:end].strip())

            if end >= text_len:
                break

            # Compute next start ensuring forward progress
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = start + max(1, chunk_size - chunk_overlap)
            start = next_start

        return chunks
