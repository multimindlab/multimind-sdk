"""
Advanced retriever module with multiple retrieval strategies.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from datetime import datetime
import networkx as nx
from ..models.base import BaseLLM
from .embedding import EmbeddingModel, EmbeddingConfig
from .vector_store import VectorStore, VectorStoreConfig, SearchResult

@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    strategy: str
    top_k: int
    similarity_threshold: float
    rerank_top_k: int
    max_hops: int
    fusion_weight: float
    custom_params: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Retrieval result with metadata."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]
    strategy: str
    hops: Optional[int]
    fusion_scores: Optional[List[float]]

class RetrievalStrategy(Enum):
    """Types of retrieval strategies."""
    HYBRID = "hybrid"
    MULTI_HOP = "multi_hop"
    FUSION = "fusion"
    GRAPH = "graph"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    DOMAIN = "domain"

class Retriever:
    """Advanced retriever with multiple retrieval strategies."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm: Optional[BaseLLM] = None,
        config: Optional[RetrievalConfig] = None,
        **kwargs
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_model: Embedding model instance
            llm: Optional LLM for advanced strategies
            config: Optional retrieval configuration
            **kwargs: Additional parameters
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.config = config or self._get_default_config()
        self.kwargs = kwargs
        
        # Initialize graph for graph-based retrieval
        if self.config.strategy == RetrievalStrategy.GRAPH.value:
            self.graph = nx.DiGraph()
        
        # Initialize temporal index
        if self.config.strategy == RetrievalStrategy.TEMPORAL.value:
            self.temporal_index = {}
        
        # Initialize domain index
        if self.config.strategy == RetrievalStrategy.DOMAIN.value:
            self.domain_index = {}

    def _get_default_config(self) -> RetrievalConfig:
        """Get default retrieval configuration."""
        return RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID.value,
            top_k=10,
            similarity_threshold=0.7,
            rerank_top_k=50,
            max_hops=3,
            fusion_weight=0.5,
            custom_params={}
        )

    async def retrieve(
        self,
        query: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve documents based on query.
        
        Args:
            query: Query string
            filter_criteria: Optional filtering criteria
            **kwargs: Additional parameters
            
        Returns:
            Retrieval result
        """
        # Generate query embedding
        query_embedding = await self.embedding_model.generate_embedding(
            query,
            EmbeddingConfig(
                model_name=self.embedding_model.model_name,
                model_type=self.embedding_model.model_type.value,
                batch_size=1,
                max_length=512,
                normalize=True,
                device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                cache_dir=None,
                custom_params={}
            )
        )
        
        # Retrieve based on strategy
        if self.config.strategy == RetrievalStrategy.HYBRID.value:
            return await self._hybrid_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.MULTI_HOP.value:
            return await self._multi_hop_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.FUSION.value:
            return await self._fusion_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.GRAPH.value:
            return await self._graph_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.HIERARCHICAL.value:
            return await self._hierarchical_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.TEMPORAL.value:
            return await self._temporal_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        elif self.config.strategy == RetrievalStrategy.DOMAIN.value:
            return await self._domain_retrieve(
                query,
                query_embedding,
                filter_criteria,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported retrieval strategy: {self.config.strategy}")

    async def _hybrid_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Hybrid retrieval combining semantic and keyword search."""
        # Get semantic search results
        semantic_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        # Get keyword search results if available
        keyword_results = []
        if self.vector_store.config.index_type == "redis":
            # Use Redis search for keyword matching
            keyword_results = await self._keyword_search(
                query,
                k=self.config.rerank_top_k,
                filter_criteria=filter_criteria
            )
        
        # Combine and rerank results
        all_results = semantic_results + keyword_results
        reranked_results = await self._rerank_results(
            query,
            all_results,
            k=self.config.top_k
        )
        
        return RetrievalResult(
            documents=[r.document for r in reranked_results],
            scores=[r.score for r in reranked_results],
            metadata={
                "strategy": "hybrid",
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results)
            },
            strategy=RetrievalStrategy.HYBRID.value,
            hops=None,
            fusion_scores=None
        )

    async def _multi_hop_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Multi-hop retrieval for complex queries."""
        if not self.llm:
            raise ValueError("LLM required for multi-hop retrieval")
        
        # Decompose query into sub-queries
        sub_queries = await self._decompose_query(query)
        
        # Initialize results
        all_results = []
        seen_ids = set()
        
        # Process each hop
        for hop in range(self.config.max_hops):
            hop_results = []
            
            # Process each sub-query
            for sub_query in sub_queries:
                # Generate embedding for sub-query
                sub_embedding = await self.embedding_model.generate_embedding(
                    sub_query,
                    EmbeddingConfig(
                        model_name=self.embedding_model.model_name,
                        model_type=self.embedding_model.model_type.value,
                        batch_size=1,
                        max_length=512,
                        normalize=True,
                        device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                        cache_dir=None,
                        custom_params={}
                    )
                )
                
                # Search for relevant documents
                results = await self.vector_store.search(
                    sub_embedding,
                    k=self.config.top_k,
                    filter_criteria=filter_criteria
                )
                
                # Add new results
                for result in results:
                    if result.id not in seen_ids:
                        hop_results.append(result)
                        seen_ids.add(result.id)
            
            # Update sub-queries based on retrieved documents
            if hop < self.config.max_hops - 1:
                sub_queries = await self._generate_follow_up_queries(
                    query,
                    [r.document for r in hop_results]
                )
            
            all_results.extend(hop_results)
        
        # Rerank final results
        reranked_results = await self._rerank_results(
            query,
            all_results,
            k=self.config.top_k
        )
        
        return RetrievalResult(
            documents=[r.document for r in reranked_results],
            scores=[r.score for r in reranked_results],
            metadata={
                "strategy": "multi_hop",
                "hops": self.config.max_hops,
                "sub_queries": sub_queries
            },
            strategy=RetrievalStrategy.MULTI_HOP.value,
            hops=self.config.max_hops,
            fusion_scores=None
        )

    async def _fusion_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """RAG-Fusion retrieval combining multiple retrieval strategies."""
        # Get results from different strategies
        semantic_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        keyword_results = await self._keyword_search(
            query,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        if self.llm:
            llm_results = await self._llm_guided_search(
                query,
                k=self.config.rerank_top_k,
                filter_criteria=filter_criteria
            )
        else:
            llm_results = []
        
        # Combine results with fusion
        all_results = []
        seen_ids = set()
        
        # Add semantic results
        for result in semantic_results:
            if result.id not in seen_ids:
                all_results.append((result, 1.0))  # Full weight for semantic
                seen_ids.add(result.id)
        
        # Add keyword results with reduced weight
        for result in keyword_results:
            if result.id not in seen_ids:
                all_results.append((result, self.config.fusion_weight))
                seen_ids.add(result.id)
        
        # Add LLM results with custom weight
        for result in llm_results:
            if result.id not in seen_ids:
                all_results.append((result, 0.8))  # High weight for LLM
                seen_ids.add(result.id)
        
        # Calculate fusion scores
        fusion_scores = []
        for result, weight in all_results:
            # Combine original score with strategy weight
            fusion_score = result.score * weight
            fusion_scores.append(fusion_score)
        
        # Sort by fusion scores
        sorted_results = [
            result for _, (result, _) in sorted(
                zip(fusion_scores, all_results),
                key=lambda x: x[0],
                reverse=True
            )
        ][:self.config.top_k]
        
        return RetrievalResult(
            documents=[r.document for r in sorted_results],
            scores=[r.score for r in sorted_results],
            metadata={
                "strategy": "fusion",
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results),
                "llm_count": len(llm_results)
            },
            strategy=RetrievalStrategy.FUSION.value,
            hops=None,
            fusion_scores=fusion_scores[:self.config.top_k]
        )

    async def _graph_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Graph-based retrieval using document relationships."""
        # Get initial results
        initial_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        # Build subgraph around initial results
        subgraph = self.graph.subgraph(
            [r.id for r in initial_results]
        ).copy()
        
        # Add edges based on document relationships
        for result in initial_results:
            # Add edges based on metadata relationships
            for key, value in result.metadata.items():
                if isinstance(value, (list, tuple)):
                    for v in value:
                        edge_key = f"{key}:{v}"
                        if edge_key in self.graph:
                            subgraph.add_edge(result.id, edge_key)
        
        # Find important nodes using PageRank
        pagerank = nx.pagerank(subgraph)
        
        # Sort results by PageRank score
        sorted_results = sorted(
            initial_results,
            key=lambda x: pagerank.get(x.id, 0),
            reverse=True
        )[:self.config.top_k]
        
        return RetrievalResult(
            documents=[r.document for r in sorted_results],
            scores=[pagerank.get(r.id, 0) for r in sorted_results],
            metadata={
                "strategy": "graph",
                "graph_size": len(subgraph),
                "pagerank_scores": pagerank
            },
            strategy=RetrievalStrategy.GRAPH.value,
            hops=None,
            fusion_scores=None
        )

    async def _hierarchical_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Hierarchical retrieval using document structure."""
        # Get initial results
        initial_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        # Score results based on hierarchy
        scored_results = []
        for result in initial_results:
            # Calculate hierarchy score
            hierarchy_score = self._calculate_hierarchy_score(
                result.metadata,
                query
            )
            
            # Combine with similarity score
            final_score = (
                result.score * 0.7 +  # Similarity weight
                hierarchy_score * 0.3  # Hierarchy weight
            )
            
            scored_results.append((result, final_score))
        
        # Sort by final score
        sorted_results = [
            result for result, _ in sorted(
                scored_results,
                key=lambda x: x[1],
                reverse=True
            )
        ][:self.config.top_k]
        
        return RetrievalResult(
            documents=[r.document for r in sorted_results],
            scores=[r.score for r in sorted_results],
            metadata={
                "strategy": "hierarchical",
                "hierarchy_scores": [score for _, score in scored_results]
            },
            strategy=RetrievalStrategy.HIERARCHICAL.value,
            hops=None,
            fusion_scores=None
        )

    async def _temporal_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Temporal-aware retrieval considering document timestamps."""
        # Get initial results
        initial_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        # Score results based on temporal relevance
        scored_results = []
        for result in initial_results:
            # Calculate temporal score
            temporal_score = self._calculate_temporal_score(
                result.metadata,
                query
            )
            
            # Combine with similarity score
            final_score = (
                result.score * 0.7 +  # Similarity weight
                temporal_score * 0.3  # Temporal weight
            )
            
            scored_results.append((result, final_score))
        
        # Sort by final score
        sorted_results = [
            result for result, _ in sorted(
                scored_results,
                key=lambda x: x[1],
                reverse=True
            )
        ][:self.config.top_k]
        
        return RetrievalResult(
            documents=[r.document for r in sorted_results],
            scores=[r.score for r in sorted_results],
            metadata={
                "strategy": "temporal",
                "temporal_scores": [score for _, score in scored_results]
            },
            strategy=RetrievalStrategy.TEMPORAL.value,
            hops=None,
            fusion_scores=None
        )

    async def _domain_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        filter_criteria: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """Domain-aware retrieval considering document domains."""
        # Get initial results
        initial_results = await self.vector_store.search(
            query_embedding,
            k=self.config.rerank_top_k,
            filter_criteria=filter_criteria
        )
        
        # Score results based on domain relevance
        scored_results = []
        for result in initial_results:
            # Calculate domain score
            domain_score = self._calculate_domain_score(
                result.metadata,
                query
            )
            
            # Combine with similarity score
            final_score = (
                result.score * 0.7 +  # Similarity weight
                domain_score * 0.3  # Domain weight
            )
            
            scored_results.append((result, final_score))
        
        # Sort by final score
        sorted_results = [
            result for result, _ in sorted(
                scored_results,
                key=lambda x: x[1],
                reverse=True
            )
        ][:self.config.top_k]
        
        return RetrievalResult(
            documents=[r.document for r in sorted_results],
            scores=[r.score for r in sorted_results],
            metadata={
                "strategy": "domain",
                "domain_scores": [score for _, score in scored_results]
            },
            strategy=RetrievalStrategy.DOMAIN.value,
            hops=None,
            fusion_scores=None
        )

    async def _keyword_search(
        self,
        query: str,
        k: int,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform keyword search."""
        if self.vector_store.config.index_type != "redis":
            return []
        
        # Build keyword query
        keywords = query.lower().split()
        keyword_query = " ".join(f"@content:{keyword}" for keyword in keywords)
        
        if filter_criteria:
            filter_query = " ".join(
                f"@tags:{{{tag}}}" for tag in filter_criteria.get("tags", [])
            )
            if filter_query:
                keyword_query = f"({filter_query}) ({keyword_query})"
        
        # Execute search
        results = []
        for doc in self.vector_store.redis_client.ft("idx").search(
            keyword_query,
            limit=k
        ).docs:
            results.append(
                SearchResult(
                    id=doc.id,
                    score=float(doc.score) if hasattr(doc, "score") else 1.0,
                    vector=[],
                    metadata=json.loads(doc.metadata),
                    document=json.loads(doc.document)
                )
            )
        
        return results

    async def _llm_guided_search(
        self,
        query: str,
        k: int,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform LLM-guided search."""
        if not self.llm:
            return []
        
        # Generate search query using LLM
        prompt = f"""
        Given the following query, generate a search query that will help find relevant documents.
        Focus on key concepts and important terms.
        
        Query: {query}
        
        Search query:
        """
        
        search_query = await self.llm.generate(prompt)
        
        # Generate embedding for search query
        search_embedding = await self.embedding_model.generate_embedding(
            search_query,
            EmbeddingConfig(
                model_name=self.embedding_model.model_name,
                model_type=self.embedding_model.model_type.value,
                batch_size=1,
                max_length=512,
                normalize=True,
                device="cuda" if self.embedding_model.device == "cuda" else "cpu",
                cache_dir=None,
                custom_params={}
            )
        )
        
        # Search using generated embedding
        return await self.vector_store.search(
            search_embedding,
            k=k,
            filter_criteria=filter_criteria
        )

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries."""
        if not self.llm:
            return [query]
        
        prompt = f"""
        Given the following complex query, break it down into simpler sub-queries that can be answered independently.
        Each sub-query should focus on a specific aspect of the original query.
        
        Complex query: {query}
        
        Sub-queries:
        1.
        """
        
        response = await self.llm.generate(prompt)
        
        # Parse sub-queries from response
        sub_queries = [
            line.strip().split(". ", 1)[1]
            for line in response.split("\n")
            if line.strip() and ". " in line
        ]
        
        return sub_queries or [query]

    async def _generate_follow_up_queries(
        self,
        original_query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up queries based on retrieved documents."""
        if not self.llm:
            return [original_query]
        
        # Format retrieved documents
        docs_text = "\n\n".join(
            f"Document {i+1}:\n{doc.get('content', '')}"
            for i, doc in enumerate(retrieved_docs)
        )
        
        prompt = f"""
        Given the original query and retrieved documents, generate follow-up queries to find more relevant information.
        Focus on aspects that haven't been fully addressed by the current documents.
        
        Original query: {original_query}
        
        Retrieved documents:
        {docs_text}
        
        Follow-up queries:
        1.
        """
        
        response = await self.llm.generate(prompt)
        
        # Parse follow-up queries from response
        follow_up_queries = [
            line.strip().split(". ", 1)[1]
            for line in response.split("\n")
            if line.strip() and ". " in line
        ]
        
        return follow_up_queries or [original_query]

    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        k: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder if available."""
        if not self.llm or len(results) <= k:
            return results[:k]
        
        # Generate reranking scores using LLM
        reranked_results = []
        for result in results:
            prompt = f"""
            Given the query and document, score the relevance of the document to the query on a scale of 0 to 1.
            Consider both semantic similarity and information relevance.
            
            Query: {query}
            
            Document:
            {result.document.get('content', '')}
            
            Relevance score (0-1):
            """
            
            score_text = await self.llm.generate(prompt)
            try:
                score = float(score_text.strip())
                reranked_results.append((result, score))
            except ValueError:
                reranked_results.append((result, result.score))
        
        # Sort by reranking score
        sorted_results = [
            result for result, _ in sorted(
                reranked_results,
                key=lambda x: x[1],
                reverse=True
            )
        ][:k]
        
        return sorted_results

    def _calculate_hierarchy_score(
        self,
        metadata: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate hierarchy score for document."""
        # Placeholder implementation
        # In practice, this would consider document structure, headings, etc.
        return 1.0

    def _calculate_temporal_score(
        self,
        metadata: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate temporal score for document."""
        # Placeholder implementation
        # In practice, this would consider document timestamps, temporal context, etc.
        return 1.0

    def _calculate_domain_score(
        self,
        metadata: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate domain score for document."""
        # Placeholder implementation
        # In practice, this would consider document domain, topic, etc.
        return 1.0 