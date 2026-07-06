"""
Advanced evaluation system for RAG with comprehensive metrics and analysis.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from transformers import AutoModel, AutoTokenizer

from ..models.base import BaseLLM
from .metrics import (
    ndcg_score,
    parse_index_list,
    parse_judge_score,
    reciprocal_rank,
    rouge_scores,
    sentence_bleu,
    tokenize,
)

logger = logging.getLogger(__name__)


def _doc_id(doc: Dict[str, Any]) -> str:
    # Docs without an explicit id are identified by a stable content hash
    if "id" in doc:
        return str(doc["id"])
    payload = doc.get("content", json.dumps(doc, sort_keys=True, default=str))
    return hashlib.sha256(str(payload).encode()).hexdigest()[:16]


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""

    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    retrieval_ndcg: float
    retrieval_mrr: float

    # Generation metrics
    generation_bleu: float
    generation_rouge: Dict[str, float]
    generation_meteor: float
    generation_bertscore: float

    # Faithfulness metrics
    faithfulness_score: float
    hallucination_score: float
    factuality_score: float
    consistency_score: float

    # Context metrics
    context_relevance: float
    context_coverage: float
    context_density: float

    # Performance metrics
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    token_usage: Dict[str, int]

    # Quality metrics
    answer_relevance: float
    answer_completeness: float
    answer_coherence: float
    answer_fluency: float

    # Custom metrics
    custom_metrics: Dict[str, float]


class EvaluationType(Enum):
    """Types of evaluation."""

    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    FAITHFULNESS = "faithfulness"
    CONTEXT = "context"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COMPREHENSIVE = "comprehensive"


@dataclass
class EvaluationResult:
    """Result of RAG evaluation."""

    metrics: EvaluationMetrics
    evaluation_type: EvaluationType
    timestamp: float
    query: str
    retrieved_documents: List[Dict[str, Any]]
    generated_response: str
    ground_truth: Optional[str]
    metadata: Dict[str, Any]


class AdvancedEvaluator:
    """Advanced evaluation system for RAG."""

    def __init__(
        self,
        model: BaseLLM,
        tokenizer: Optional[AutoTokenizer] = None,
        embedding_model: Optional[AutoModel] = None,
        **kwargs,
    ):
        """
        Initialize advanced evaluator.

        Args:
            model: Language model for evaluation
            tokenizer: Optional tokenizer for metrics
            embedding_model: Optional embedding model for semantic metrics
            **kwargs: Additional parameters
        """
        self.model = model
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        self.embedding_model = embedding_model or AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.kwargs = kwargs

    async def evaluate(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        ground_truth: Optional[str] = None,
        evaluation_type: EvaluationType = EvaluationType.COMPREHENSIVE,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate RAG system performance.

        Args:
            query: User query
            retrieved_documents: Retrieved documents
            generated_response: Generated response
            ground_truth: Optional ground truth
            evaluation_type: Type of evaluation to perform
            **kwargs: Additional parameters

        Returns:
            Evaluation result
        """
        start_time = datetime.now()

        # Initialize metrics
        metrics = EvaluationMetrics(
            retrieval_precision=0.0,
            retrieval_recall=0.0,
            retrieval_f1=0.0,
            retrieval_ndcg=0.0,
            retrieval_mrr=0.0,
            generation_bleu=0.0,
            generation_rouge={},
            generation_meteor=0.0,
            generation_bertscore=0.0,
            faithfulness_score=0.0,
            hallucination_score=0.0,
            factuality_score=0.0,
            consistency_score=0.0,
            context_relevance=0.0,
            context_coverage=0.0,
            context_density=0.0,
            retrieval_latency=0.0,
            generation_latency=0.0,
            total_latency=0.0,
            token_usage={},
            answer_relevance=0.0,
            answer_completeness=0.0,
            answer_coherence=0.0,
            answer_fluency=0.0,
            custom_metrics={},
        )

        # Perform evaluation based on type; a failing phase is recorded, not silently zeroed
        phases = [
            (
                EvaluationType.RETRIEVAL,
                lambda: self._evaluate_retrieval(query, retrieved_documents, metrics, **kwargs),
            ),
            (
                EvaluationType.GENERATION,
                lambda: self._evaluate_generation(
                    generated_response, ground_truth, metrics, **kwargs
                ),
            ),
            (
                EvaluationType.FAITHFULNESS,
                lambda: self._evaluate_faithfulness(
                    query, retrieved_documents, generated_response, metrics, **kwargs
                ),
            ),
            (
                EvaluationType.CONTEXT,
                lambda: self._evaluate_context(
                    query, retrieved_documents, generated_response, metrics, **kwargs
                ),
            ),
            (
                EvaluationType.QUALITY,
                lambda: self._evaluate_quality(query, generated_response, metrics, **kwargs),
            ),
        ]
        errors: Dict[str, str] = {}
        for phase_type, run_phase in phases:
            if evaluation_type not in [phase_type, EvaluationType.COMPREHENSIVE]:
                continue
            try:
                await run_phase()
            except Exception as e:
                logger.warning("%s evaluation failed: %s", phase_type.value, e)
                errors[phase_type.value] = str(e)

        # Calculate performance metrics
        end_time = datetime.now()
        metrics.total_latency = (end_time - start_time).total_seconds()

        metadata = dict(kwargs)
        if errors:
            metadata["errors"] = errors

        return EvaluationResult(
            metrics=metrics,
            evaluation_type=evaluation_type,
            timestamp=datetime.now().timestamp(),
            query=query,
            retrieved_documents=retrieved_documents,
            generated_response=generated_response,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    async def _evaluate_retrieval(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        metrics: EvaluationMetrics,
        **kwargs,
    ) -> None:
        """Evaluate retrieval performance."""
        # Calculate precision and recall
        relevant_docs = await self._get_relevant_documents(query, **kwargs)
        retrieved_doc_ids = [_doc_id(doc) for doc in retrieved_documents]
        relevant_doc_ids = [_doc_id(doc) for doc in relevant_docs]

        # Calculate metrics
        metrics.retrieval_precision = (
            len(set(retrieved_doc_ids) & set(relevant_doc_ids)) / len(retrieved_doc_ids)
            if retrieved_doc_ids
            else 0.0
        )

        metrics.retrieval_recall = (
            len(set(retrieved_doc_ids) & set(relevant_doc_ids)) / len(relevant_doc_ids)
            if relevant_doc_ids
            else 0.0
        )

        metrics.retrieval_f1 = (
            2
            * metrics.retrieval_precision
            * metrics.retrieval_recall
            / (metrics.retrieval_precision + metrics.retrieval_recall)
            if (metrics.retrieval_precision + metrics.retrieval_recall) > 0
            else 0.0
        )

        # Calculate NDCG
        metrics.retrieval_ndcg = await self._calculate_ndcg(retrieved_documents, relevant_docs)

        # Calculate MRR
        metrics.retrieval_mrr = await self._calculate_mrr(retrieved_documents, relevant_docs)

    async def _evaluate_generation(
        self,
        generated_response: str,
        ground_truth: Optional[str],
        metrics: EvaluationMetrics,
        **kwargs,
    ) -> None:
        """Evaluate generation performance."""
        if not ground_truth:
            return

        # Calculate BLEU score
        metrics.generation_bleu = await self._calculate_bleu(generated_response, ground_truth)

        # Calculate ROUGE scores
        metrics.generation_rouge = await self._calculate_rouge(generated_response, ground_truth)

        # Calculate METEOR score
        metrics.generation_meteor = await self._calculate_meteor(generated_response, ground_truth)

        # Calculate BERTScore
        metrics.generation_bertscore = await self._calculate_bertscore(
            generated_response, ground_truth
        )

    async def _evaluate_faithfulness(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        metrics: EvaluationMetrics,
        **kwargs,
    ) -> None:
        """Evaluate faithfulness of generated response."""
        # Check for hallucinations
        metrics.hallucination_score = await self._detect_hallucinations(
            query, retrieved_documents, generated_response, **kwargs
        )

        # Check factuality
        metrics.factuality_score = await self._check_factuality(
            query, retrieved_documents, generated_response, **kwargs
        )

        # Check consistency
        metrics.consistency_score = await self._check_consistency(
            query, generated_response, **kwargs
        )

        # Calculate overall faithfulness
        metrics.faithfulness_score = (
            0.4 * (1 - metrics.hallucination_score)
            + 0.4 * metrics.factuality_score
            + 0.2 * metrics.consistency_score
        )

    async def _evaluate_context(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        metrics: EvaluationMetrics,
        **kwargs,
    ) -> None:
        """Evaluate context usage."""
        # Calculate context relevance
        metrics.context_relevance = await self._calculate_context_relevance(
            query, retrieved_documents, generated_response, **kwargs
        )

        # Calculate context coverage
        metrics.context_coverage = await self._calculate_context_coverage(
            retrieved_documents, generated_response, **kwargs
        )

        # Calculate context density
        metrics.context_density = await self._calculate_context_density(
            retrieved_documents, generated_response, **kwargs
        )

    async def _evaluate_quality(
        self, query: str, generated_response: str, metrics: EvaluationMetrics, **kwargs
    ) -> None:
        """Evaluate response quality."""
        # Calculate answer relevance
        metrics.answer_relevance = await self._calculate_answer_relevance(
            query, generated_response, **kwargs
        )

        # Calculate answer completeness
        metrics.answer_completeness = await self._calculate_answer_completeness(
            query, generated_response, **kwargs
        )

        # Calculate answer coherence
        metrics.answer_coherence = await self._calculate_answer_coherence(
            generated_response, **kwargs
        )

        # Calculate answer fluency
        metrics.answer_fluency = await self._calculate_answer_fluency(generated_response, **kwargs)

    async def _get_relevant_documents(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Get relevant documents for evaluation."""
        all_documents = kwargs.get("all_documents", [])
        if not all_documents:
            return []

        # Use LLM to determine relevance
        prompt = f"""
        Determine which of the following documents are relevant to the query.
        Consider:
        1. Topic relevance
        2. Information value
        3. Query coverage

        Query: {query}

        Documents (0-indexed):
        {all_documents}

        Respond with only a JSON list of the 0-based indices of the relevant documents, e.g. [0, 2].
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        indices = parse_index_list(response)
        return [all_documents[i] for i in indices if 0 <= i < len(all_documents)]

    async def _calculate_ndcg(
        self, retrieved_documents: List[Dict[str, Any]], relevant_documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate NDCG score."""
        if not retrieved_documents or not relevant_documents:
            return 0.0
        # Graded relevance when documents carry a "relevance" field, binary otherwise
        gains_by_id = {_doc_id(doc): float(doc.get("relevance", 1.0)) for doc in relevant_documents}
        gains = [gains_by_id.get(_doc_id(doc), 0.0) for doc in retrieved_documents]
        return ndcg_score(
            gains, k=len(retrieved_documents), ideal_relevances=list(gains_by_id.values())
        )

    async def _calculate_mrr(
        self, retrieved_documents: List[Dict[str, Any]], relevant_documents: List[Dict[str, Any]]
    ) -> float:
        """Calculate MRR score."""
        relevant_ids = {_doc_id(doc) for doc in relevant_documents}
        return reciprocal_rank(
            [1.0 if _doc_id(doc) in relevant_ids else 0.0 for doc in retrieved_documents]
        )

    async def _calculate_bleu(self, generated: str, reference: str) -> float:
        """Calculate BLEU score."""
        return sentence_bleu(generated, reference)

    async def _calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        return rouge_scores(generated, reference)

    async def _calculate_meteor(self, generated: str, reference: str) -> float:
        """Calculate METEOR score."""
        try:
            from nltk.translate.meteor_score import meteor_score
        except ImportError as e:
            raise NotImplementedError(
                "METEOR requires the optional 'nltk' package with WordNet data. "
                "Install with: pip install nltk && python -m nltk.downloader wordnet omw-1.4"
            ) from e
        return float(meteor_score([tokenize(reference)], tokenize(generated)))

    async def _calculate_bertscore(self, generated: str, reference: str) -> float:
        """Calculate BERTScore."""
        try:
            from bert_score import score as bert_score
        except ImportError as e:
            raise NotImplementedError(
                "BERTScore requires the optional 'bert-score' package. "
                "Install with: pip install bert-score"
            ) from e
        _, _, f1 = bert_score([generated], [reference], lang="en")
        return float(f1.mean())

    async def _detect_hallucinations(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        **kwargs,
    ) -> float:
        """Detect hallucinations in generated response."""
        # Use LLM to detect hallucinations
        prompt = f"""
        Detect hallucinations in the generated response.
        Consider:
        1. Information not present in retrieved documents
        2. Contradictions with retrieved documents
        3. Fabricated details

        Query: {query}

        Retrieved Documents:
        {retrieved_documents}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 0 means fully grounded and 1 means fully hallucinated.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _check_factuality(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        **kwargs,
    ) -> float:
        """Check factuality of generated response."""
        # Use LLM to check factuality
        prompt = f"""
        Check factuality of the generated response.
        Consider:
        1. Accuracy of facts
        2. Source attribution
        3. Information consistency

        Query: {query}

        Retrieved Documents:
        {retrieved_documents}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully factual.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _check_consistency(self, query: str, generated_response: str, **kwargs) -> float:
        """Check consistency of generated response."""
        # Use LLM to check consistency
        prompt = f"""
        Check consistency of the generated response.
        Consider:
        1. Internal consistency
        2. Logical flow
        3. Argument coherence

        Query: {query}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully consistent.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_context_relevance(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        generated_response: str,
        **kwargs,
    ) -> float:
        """Calculate context relevance score."""
        # Use LLM to calculate context relevance
        prompt = f"""
        Calculate relevance of retrieved context to the query.
        Consider:
        1. Query coverage
        2. Information relevance
        3. Context utilization

        Query: {query}

        Retrieved Documents:
        {retrieved_documents}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully relevant.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_context_coverage(
        self, retrieved_documents: List[Dict[str, Any]], generated_response: str, **kwargs
    ) -> float:
        """Calculate context coverage score."""
        # Use LLM to calculate context coverage
        prompt = f"""
        Calculate coverage of retrieved context in the response.
        Consider:
        1. Information utilization
        2. Context completeness
        3. Detail preservation

        Retrieved Documents:
        {retrieved_documents}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means full coverage.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_context_density(
        self, retrieved_documents: List[Dict[str, Any]], generated_response: str, **kwargs
    ) -> float:
        """Calculate context density score."""
        # Use LLM to calculate context density
        prompt = f"""
        Calculate density of information from context in the response.
        Consider:
        1. Information concentration
        2. Detail level
        3. Context efficiency

        Retrieved Documents:
        {retrieved_documents}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means maximally dense.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_answer_relevance(
        self, query: str, generated_response: str, **kwargs
    ) -> float:
        """Calculate answer relevance score."""
        # Use LLM to calculate answer relevance
        prompt = f"""
        Calculate relevance of the answer to the query.
        Consider:
        1. Query addressing
        2. Information relevance
        3. Answer focus

        Query: {query}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully relevant.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_answer_completeness(
        self, query: str, generated_response: str, **kwargs
    ) -> float:
        """Calculate answer completeness score."""
        # Use LLM to calculate answer completeness
        prompt = f"""
        Calculate completeness of the answer.
        Consider:
        1. Query coverage
        2. Information completeness
        3. Detail sufficiency

        Query: {query}

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully complete.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_answer_coherence(self, generated_response: str, **kwargs) -> float:
        """Calculate answer coherence score."""
        # Use LLM to calculate answer coherence
        prompt = f"""
        Calculate coherence of the answer.
        Consider:
        1. Logical flow
        2. Structure clarity
        3. Argument coherence

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully coherent.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)

    async def _calculate_answer_fluency(self, generated_response: str, **kwargs) -> float:
        """Calculate answer fluency score."""
        # Use LLM to calculate answer fluency
        prompt = f"""
        Calculate fluency of the answer.
        Consider:
        1. Language quality
        2. Grammar correctness
        3. Expression clarity

        Generated Response:
        {generated_response}

        Respond with only a single number between 0 and 1, where 1 means fully fluent.
        """

        response = await self.model.generate(prompt=prompt, **kwargs)
        return parse_judge_score(response)
