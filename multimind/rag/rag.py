"""
Enhanced RAG implementation with advanced features.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
import logging
from datetime import datetime
import numpy as np

from .base import (
    BaseRAG, RAGError, DocumentProcessingError,
    RetrievalError, GenerationError, RetrievalStrategy,
    ChunkingStrategy, RetrievalMetrics, GenerationMetrics
)
from .document_loader import DocumentLoader, LoadedDocument
from .document_processor import EnhancedDocumentProcessor
from .retrieval import (
    HybridRetriever, QueryDecomposer, MultiVectorRetriever,
    RetrievalResult
)
from .context_optimizer import (
    ContextOptimizer, PromptGenerator,
    AdvancedRAGPrompting, OptimizedContext
)
from .evaluation import RAGEvaluator, RAGEvaluation
from .memory import TokenAwareMemory, PersistentMemoryStore
from .advanced_patterns import (
    MultiHopRetriever, RAGFusion, GraphRAG,
    SelfImprovingRAG
)

logger = logging.getLogger(__name__)

@dataclass
class RAGStats:
    """Statistics for RAG system."""
    total_documents: int
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    retrieval_latency: float
    generation_latency: float
    memory_usage: Dict[str, int]
    evaluation_metrics: Optional[RAGEvaluation] = None

class EnhancedRAG(BaseRAG):
    """Enhanced RAG implementation with advanced features."""

    def __init__(
        self,
        model: Any,
        vector_store: Any,
        document_loader: Optional[DocumentLoader] = None,
        document_processor: Optional[EnhancedDocumentProcessor] = None,
        retriever: Optional[HybridRetriever] = None,
        context_optimizer: Optional[ContextOptimizer] = None,
        prompt_generator: Optional[PromptGenerator] = None,
        evaluator: Optional[RAGEvaluator] = None,
        memory: Optional[TokenAwareMemory] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        max_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize enhanced RAG system.
        
        Args:
            model: Language model
            vector_store: Vector store
            document_loader: Optional document loader
            document_processor: Optional document processor
            retriever: Optional retriever
            context_optimizer: Optional context optimizer
            prompt_generator: Optional prompt generator
            evaluator: Optional evaluator
            memory: Optional memory system
            retrieval_strategy: Retrieval strategy
            chunking_strategy: Chunking strategy
            max_tokens: Maximum tokens for context
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            vector_store=vector_store,
            retrieval_strategy=retrieval_strategy,
            chunking_strategy=chunking_strategy,
            **kwargs
        )
        
        # Initialize components
        self.document_loader = document_loader or DocumentLoader()
        self.document_processor = document_processor or EnhancedDocumentProcessor(
            model=model,
            chunking_strategy=chunking_strategy,
            **kwargs
        )
        self.retriever = retriever or HybridRetriever(
            model=model,
            vector_store=vector_store,
            **kwargs
        )
        self.context_optimizer = context_optimizer or ContextOptimizer(
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )
        self.prompt_generator = prompt_generator or PromptGenerator(
            model=model,
            **kwargs
        )
        self.evaluator = evaluator or RAGEvaluator(model=model, **kwargs)
        self.memory = memory or TokenAwareMemory(
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Initialize advanced components
        self.multi_hop = MultiHopRetriever(
            model=model,
            retriever=self.retriever,
            **kwargs
        )
        self.rag_fusion = RAGFusion(
            model=model,
            retriever=self.retriever,
            **kwargs
        )
        self.graph_rag = GraphRAG(
            model=model,
            retriever=self.retriever,
            **kwargs
        )
        self.self_improving = SelfImprovingRAG(
            model=model,
            retriever=self.retriever,
            memory=self.memory,
            **kwargs
        )
        
        # Initialize stats
        self.stats = RAGStats(
            total_documents=0,
            total_chunks=0,
            total_tokens=0,
            avg_chunk_size=0.0,
            retrieval_latency=0.0,
            generation_latency=0.0,
            memory_usage={}
        )

    async def add_documents(
        self,
        documents: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]],
        **kwargs
    ) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: Documents to add
            **kwargs: Additional parameters
        """
        try:
            # Load documents
            loaded_docs = await self.document_loader.load_documents(
                documents=documents,
                **kwargs
            )
            
            # Process documents
            processed_chunks = await self.document_processor.process_documents(
                documents=loaded_docs,
                **kwargs
            )
            
            # Add to vector store
            await self.vector_store.add_documents(
                documents=processed_chunks,
                **kwargs
            )
            
            # Add to graph RAG if using
            if self.retrieval_strategy == RetrievalStrategy.GRAPH:
                for doc in loaded_docs:
                    await self.graph_rag.add_document(
                        doc=doc,
                        **kwargs
                    )
            
            # Update stats
            self.stats.total_documents += len(loaded_docs)
            self.stats.total_chunks += len(processed_chunks)
            self.stats.total_tokens += sum(
                chunk.get("tokens", 0)
                for chunk in processed_chunks
            )
            self.stats.avg_chunk_size = (
                self.stats.total_tokens / self.stats.total_chunks
                if self.stats.total_chunks > 0
                else 0.0
            )
            
            # Add to memory if important
            for doc in loaded_docs:
                importance = await self.memory._calculate_importance(
                    content=doc["content"],
                    **kwargs
                )
                if importance > self.memory.importance_threshold:
                    await self.memory.add_to_memory(
                        content=doc["content"],
                        memory_type="long_term",
                        metadata={
                            "type": "document",
                            "importance": importance,
                            **doc.get("metadata", {})
                        }
                    )
        
        except Exception as e:
            raise DocumentProcessingError(
                f"Error processing documents: {str(e)}"
            ) from e

    async def search(
        self,
        query: str,
        k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Query to search for
            k: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of relevant documents
        """
        try:
            start_time = time.time()
            
            # Get relevant memory
            memory_items = await self.memory.get_relevant_memory(
                query=query,
                k=k,
                **kwargs
            )
            
            # Choose retrieval strategy
            if self.retrieval_strategy == RetrievalStrategy.HYBRID:
                results = await self.retriever.retrieve(
                    query=query,
                    k=k,
                    **kwargs
                )
            elif self.retrieval_strategy == RetrievalStrategy.MULTI_HOP:
                results, steps = await self.multi_hop.retrieve(
                    query=query,
                    k=k,
                    **kwargs
                )
            elif self.retrieval_strategy == RetrievalStrategy.FUSION:
                fusion_result = await self.rag_fusion.fuse(
                    query=query,
                    **kwargs
                )
                results = fusion_result.fused_results[:k]
            elif self.retrieval_strategy == RetrievalStrategy.GRAPH:
                docs, entities = await self.graph_rag.retrieve(
                    query=query,
                    **kwargs
                )
                results = docs[:k]
            else:
                results = await self.retriever.retrieve(
                    query=query,
                    k=k,
                    **kwargs
                )
            
            # Update stats
            self.stats.retrieval_latency = time.time() - start_time
            
            return results
        
        except Exception as e:
            raise RetrievalError(
                f"Error retrieving documents: {str(e)}"
            ) from e

    async def query(
        self,
        query: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query: Query to process
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response, metadata)
        """
        try:
            start_time = time.time()
            
            # Use self-improving RAG if enabled
            if kwargs.get("use_self_improving", False):
                response, metadata = await self.self_improving.process_query(
                    query=query,
                    **kwargs
                )
            else:
                # Get relevant documents
                docs = await self.search(
                    query=query,
                    **kwargs
                )
                
                # Optimize context
                optimized_context = await self.context_optimizer.optimize_context(
                    query=query,
                    chunks=docs,
                    **kwargs
                )
                
                # Generate prompt
                prompt = await self.prompt_generator.generate_prompt(
                    query=query,
                    context=optimized_context,
                    **kwargs
                )
                
                # Generate response
                response = await self.model.generate(
                    prompt=prompt,
                    **kwargs
                )
                
                # Evaluate response
                evaluation = await self.evaluator.evaluate_rag(
                    query=query,
                    response=response,
                    context=docs,
                    **kwargs
                )
                
                metadata = {
                    "evaluation": evaluation,
                    "context": optimized_context,
                    "prompt": prompt
                }
            
            # Update stats
            self.stats.generation_latency = time.time() - start_time
            self.stats.evaluation_metrics = evaluation
            
            return response, metadata
        
        except Exception as e:
            raise GenerationError(
                f"Error generating response: {str(e)}"
            ) from e

    async def evaluate(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> RAGEvaluation:
        """
        Evaluate RAG system performance.
        
        Args:
            query: Query used
            response: Generated response
            context: Retrieved context
            **kwargs: Additional parameters
            
        Returns:
            Evaluation results
        """
        try:
            return await self.evaluator.evaluate_rag(
                query=query,
                response=response,
                context=context,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error evaluating RAG: {str(e)}")
            return None

    async def validate_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Validate documents before processing.
        
        Args:
            documents: Documents to validate
            **kwargs: Additional parameters
            
        Returns:
            List of valid documents
        """
        try:
            valid_docs = []
            for doc in documents:
                # Check required fields
                if not all(k in doc for k in ["content", "id"]):
                    logger.warning(f"Document missing required fields: {doc}")
                    continue
                
                # Validate content
                if not doc["content"].strip():
                    logger.warning(f"Document has empty content: {doc['id']}")
                    continue
                
                # Add validation metadata
                doc["metadata"] = {
                    **doc.get("metadata", {}),
                    "validated_at": datetime.now().isoformat(),
                    "validation_status": "valid"
                }
                
                valid_docs.append(doc)
            
            return valid_docs
        
        except Exception as e:
            logger.error(f"Error validating documents: {str(e)}")
            return []

    async def reindex(
        self,
        **kwargs
    ) -> None:
        """
        Reindex all documents.
        
        Args:
            **kwargs: Additional parameters
        """
        try:
            # Get all documents
            all_docs = await self.vector_store.get_all_documents(**kwargs)
            
            # Clear vector store
            await self.vector_store.clear(**kwargs)
            
            # Reprocess and add documents
            await self.add_documents(
                documents=all_docs,
                **kwargs
            )
        
        except Exception as e:
            logger.error(f"Error reindexing: {str(e)}")
            raise

    def get_stats(self) -> RAGStats:
        """
        Get RAG system statistics.
        
        Returns:
            System statistics
        """
        return self.stats