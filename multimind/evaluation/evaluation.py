"""
Comprehensive evaluation system for RAG components.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from ..models.base import BaseLLM

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    precision: float
    recall: float
    f1_score: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    relevance_scores: List[float]
    latency_ms: float

@dataclass
class GenerationMetrics:
    """Metrics for generation quality."""
    answer_relevance: float
    faithfulness: float
    hallucination_score: float
    coherence: float
    fluency: float
    latency_ms: float
    token_usage: Dict[str, int]

@dataclass
class RAGEvaluation:
    """Complete RAG evaluation results."""
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    overall_score: float
    component_scores: Dict[str, float]

class EvaluationMetric(Enum):
    """Different evaluation metrics."""
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MRR = "mrr"
    NDCG = "ndcg"
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    COHERENCE = "coherence"
    FLUENCY = "fluency"

class RAGEvaluator:
    """Evaluates RAG system components and overall performance."""

    def __init__(
        self,
        model: BaseLLM,
        cross_encoder: Optional[CrossEncoder] = None,
        **kwargs
    ):
        self.model = model
        self.cross_encoder = cross_encoder
        self.kwargs = kwargs

    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Search query
            retrieved_docs: Retrieved documents
            ground_truth: Optional ground truth documents
            **kwargs: Additional evaluation parameters
            
        Returns:
            Retrieval metrics
        """
        # Calculate relevance scores
        relevance_scores = await self._calculate_relevance_scores(
            query,
            retrieved_docs
        )
        
        # Calculate precision, recall, and F1 if ground truth is provided
        precision, recall, f1 = 0.0, 0.0, 0.0
        if ground_truth:
            precision, recall, f1 = self._calculate_precision_recall_f1(
                retrieved_docs,
                ground_truth
            )
        
        # Calculate MRR and NDCG
        mrr = self._calculate_mrr(relevance_scores)
        ndcg = self._calculate_ndcg(relevance_scores)
        
        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mrr=mrr,
            ndcg=ndcg,
            relevance_scores=relevance_scores,
            latency_ms=kwargs.get("latency_ms", 0.0)
        )

    async def evaluate_generation(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            query: User query
            response: Generated response
            context: Retrieved context
            ground_truth: Optional ground truth answer
            **kwargs: Additional evaluation parameters
            
        Returns:
            Generation metrics
        """
        # Calculate answer relevance
        answer_relevance = await self._calculate_answer_relevance(
            query,
            response
        )
        
        # Calculate faithfulness
        faithfulness = await self._calculate_faithfulness(
            response,
            context
        )
        
        # Calculate hallucination score
        hallucination_score = await self._calculate_hallucination_score(
            response,
            context
        )
        
        # Calculate coherence and fluency
        coherence = await self._calculate_coherence(response)
        fluency = await self._calculate_fluency(response)
        
        return GenerationMetrics(
            answer_relevance=answer_relevance,
            faithfulness=faithfulness,
            hallucination_score=hallucination_score,
            coherence=coherence,
            fluency=fluency,
            latency_ms=kwargs.get("latency_ms", 0.0),
            token_usage=kwargs.get("token_usage", {})
        )

    async def evaluate_rag(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        response: str,
        ground_truth_docs: Optional[List[Dict[str, Any]]] = None,
        ground_truth_response: Optional[str] = None,
        **kwargs
    ) -> RAGEvaluation:
        """
        Evaluate complete RAG system.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            response: Generated response
            ground_truth_docs: Optional ground truth documents
            ground_truth_response: Optional ground truth response
            **kwargs: Additional evaluation parameters
            
        Returns:
            Complete RAG evaluation results
        """
        # Evaluate retrieval
        retrieval_metrics = await self.evaluate_retrieval(
            query,
            retrieved_docs,
            ground_truth_docs,
            **kwargs
        )
        
        # Evaluate generation
        generation_metrics = await self.evaluate_generation(
            query,
            response,
            retrieved_docs,
            ground_truth_response,
            **kwargs
        )
        
        # Calculate component scores
        component_scores = {
            "retrieval": self._calculate_component_score(retrieval_metrics),
            "generation": self._calculate_component_score(generation_metrics)
        }
        
        # Calculate overall score
        overall_score = np.mean(list(component_scores.values()))
        
        return RAGEvaluation(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            overall_score=overall_score,
            component_scores=component_scores
        )

    async def _calculate_relevance_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate relevance scores for documents."""
        if self.cross_encoder:
            # Use cross-encoder for more accurate relevance scoring
            pairs = [(query, doc["text"]) for doc in documents]
            scores = self.cross_encoder.predict(pairs)
            return [float(score) for score in scores]
        else:
            # Fallback to cosine similarity
            query_embedding = await self.model.embeddings([query])[0]
            doc_embeddings = await self.model.embeddings([doc["text"] for doc in documents])
            similarities = [
                cosine_similarity([query_embedding], [doc_emb])[0][0]
                for doc_emb in doc_embeddings
            ]
            return [float(sim) for sim in similarities]

    def _calculate_precision_recall_f1(
        self,
        retrieved: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        # Convert to sets of document IDs or content
        retrieved_set = {doc["text"] for doc in retrieved}
        ground_truth_set = {doc["text"] for doc in ground_truth}
        
        # Calculate metrics
        true_positives = len(retrieved_set & ground_truth_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    def _calculate_mrr(self, relevance_scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not relevance_scores:
            return 0.0
        
        # Find rank of first relevant document (score > 0.5)
        for i, score in enumerate(relevance_scores):
            if score > 0.5:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            dcg += (2 ** score - 1) / np.log2(i + 2)
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores[:k]):
            idcg += (2 ** score - 1) / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0

    async def _calculate_answer_relevance(
        self,
        query: str,
        response: str
    ) -> float:
        """Calculate relevance of answer to query."""
        # Generate embeddings
        query_embedding = await self.model.embeddings([query])[0]
        response_embedding = await self.model.embeddings([response])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
        return float(similarity)

    async def _calculate_faithfulness(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate faithfulness of response to context."""
        # Generate embeddings
        response_embedding = await self.model.embeddings([response])[0]
        context_embeddings = await self.model.embeddings([doc["text"] for doc in context])
        
        # Calculate average similarity to context
        similarities = [
            cosine_similarity([response_embedding], [ctx_emb])[0][0]
            for ctx_emb in context_embeddings
        ]
        return float(np.mean(similarities))

    async def _calculate_hallucination_score(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate hallucination score (1 - faithfulness)."""
        faithfulness = await self._calculate_faithfulness(response, context)
        return 1.0 - faithfulness

    async def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence of response."""
        # Split into sentences
        sentences = response.split(". ")
        if len(sentences) < 2:
            return 1.0
        
        # Generate embeddings for sentences
        sentence_embeddings = await self.model.embeddings(sentences)
        
        # Calculate average similarity between consecutive sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            similarity = cosine_similarity(
                [sentence_embeddings[i]],
                [sentence_embeddings[i + 1]]
            )[0][0]
            similarities.append(similarity)
        
        return float(np.mean(similarities))

    async def _calculate_fluency(self, response: str) -> float:
        """Calculate fluency of response."""
        # This is a placeholder implementation
        # In practice, you might want to use a language model or other metrics
        return 1.0

    def _calculate_component_score(
        self,
        metrics: Union[RetrievalMetrics, GenerationMetrics]
    ) -> float:
        """Calculate overall score for a component."""
        if isinstance(metrics, RetrievalMetrics):
            # Weight different retrieval metrics
            weights = {
                "precision": 0.3,
                "recall": 0.3,
                "f1_score": 0.2,
                "mrr": 0.1,
                "ndcg": 0.1
            }
            return sum(
                getattr(metrics, metric) * weight
                for metric, weight in weights.items()
            )
        else:
            # Weight different generation metrics
            weights = {
                "answer_relevance": 0.3,
                "faithfulness": 0.3,
                "hallucination_score": 0.1,
                "coherence": 0.15,
                "fluency": 0.15
            }
            return sum(
                getattr(metrics, metric) * weight
                for metric, weight in weights.items()
            ) 