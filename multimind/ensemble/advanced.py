"""
Advanced ensemble logic for combining results from multiple providers.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pydantic import BaseModel
from enum import Enum
import asyncio
import numpy as np
from ..core.provider import GenerationResult, EmbeddingResult, ImageAnalysisResult
from ..core.router import Router, TaskType

class EnsembleMethod(str, Enum):
    """Methods for combining ensemble results."""
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_CASCADE = "confidence_cascade"
    PARALLEL_VOTING = "parallel_voting"
    MAJORITY_VOTING = "majority_voting"
    RANK_BASED = "rank_based"

class ConfidenceScore(BaseModel):
    """Confidence score for a result."""
    score: float  # 0.0 to 1.0
    explanation: str
    metadata: Dict[str, Any] = {}

class EnsembleResult(BaseModel):
    """Result from ensemble combination."""
    result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]
    confidence: ConfidenceScore
    provider_votes: Dict[str, float]  # Provider name to vote weight
    metadata: Dict[str, Any] = {}

class AdvancedEnsemble:
    """Advanced ensemble system for combining results from multiple providers."""
    
    def __init__(self, router: Router):
        """Initialize the ensemble system."""
        self.router = router
    
    async def combine_results(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        method: EnsembleMethod,
        task_type: TaskType,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using the specified method."""
        if method == EnsembleMethod.WEIGHTED_VOTING:
            return await self._weighted_voting(results, **kwargs)
        elif method == EnsembleMethod.CONFIDENCE_CASCADE:
            return await self._confidence_cascade(results, task_type, **kwargs)
        elif method == EnsembleMethod.PARALLEL_VOTING:
            return await self._parallel_voting(results, task_type, **kwargs)
        elif method == EnsembleMethod.MAJORITY_VOTING:
            return await self._majority_voting(results, **kwargs)
        elif method == EnsembleMethod.RANK_BASED:
            return await self._rank_based(results, task_type, **kwargs)
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
    
    async def _weighted_voting(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using weighted voting."""
        if not weights:
            weights = {result.provider: 1.0 for result in results}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate weighted scores for each result
        weighted_scores = []
        for result in results:
            weight = normalized_weights.get(result.provider, 0.0)
            weighted_scores.append((result, weight))
        
        # Select result with highest weight
        best_result, best_weight = max(weighted_scores, key=lambda x: x[1])
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=best_weight,
                explanation=f"Selected result from {best_result.provider} with weight {best_weight:.2f}"
            ),
            provider_votes=normalized_weights
        )
    
    async def _confidence_cascade(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        task_type: TaskType,
        confidence_threshold: float = 0.8,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using confidence-based cascade."""
        # Evaluate confidence for each result
        confidence_scores = []
        for result in results:
            confidence = await self._evaluate_confidence(result, task_type, **kwargs)
            confidence_scores.append((result, confidence))
        
        # Sort by confidence score
        confidence_scores.sort(key=lambda x: x[1].score, reverse=True)
        
        # Find first result above threshold
        for result, confidence in confidence_scores:
            if confidence.score >= confidence_threshold:
                return EnsembleResult(
                    result=result,
                    confidence=confidence,
                    provider_votes={r.provider: c.score for r, c in confidence_scores}
                )
        
        # If no result meets threshold, return highest confidence
        best_result, best_confidence = confidence_scores[0]
        return EnsembleResult(
            result=best_result,
            confidence=best_confidence,
            provider_votes={r.provider: c.score for r, c in confidence_scores}
        )
    
    async def _parallel_voting(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        task_type: TaskType,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using parallel voting with LLM evaluator."""
        # Get LLM evaluation for each result
        evaluations = await asyncio.gather(*[
            self._evaluate_with_llm(result, task_type, **kwargs)
            for result in results
        ])
        
        # Calculate scores from evaluations
        scores = []
        for result, evaluation in zip(results, evaluations):
            score = self._parse_llm_evaluation(evaluation)
            scores.append((result, score))
        
        # Normalize scores
        total_score = sum(score for _, score in scores)
        normalized_scores = {r.provider: s/total_score for r, s in scores}
        
        # Select best result
        best_result, best_score = max(scores, key=lambda x: x[1])
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=best_score,
                explanation=f"Selected result from {best_result.provider} with LLM evaluation score {best_score:.2f}"
            ),
            provider_votes=normalized_scores
        )
    
    async def _majority_voting(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        **kwargs
    ) -> EnsembleResult:
        """Combine results using simple majority voting."""
        # Count votes for each unique result
        result_counts = {}
        for result in results:
            key = str(result.result)  # Use string representation as key
            if key not in result_counts:
                result_counts[key] = (result, 0)
            result_counts[key] = (result, result_counts[key][1] + 1)
        
        # Find result with most votes
        best_result, vote_count = max(result_counts.values(), key=lambda x: x[1])
        total_votes = len(results)
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=vote_count/total_votes,
                explanation=f"Selected result with {vote_count}/{total_votes} votes"
            ),
            provider_votes={r.provider: 1.0 for r in results}
        )
    
    async def _rank_based(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        task_type: TaskType,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using rank-based selection."""
        # Get rankings from each provider
        rankings = await asyncio.gather(*[
            self._get_provider_ranking(result, task_type, **kwargs)
            for result in results
        ])
        
        # Calculate Borda count
        borda_scores = {}
        for result, ranking in zip(results, rankings):
            score = self._calculate_borda_score(ranking, len(results))
            borda_scores[result.provider] = score
        
        # Normalize scores
        total_score = sum(borda_scores.values())
        normalized_scores = {k: v/total_score for k, v in borda_scores.items()}
        
        # Select result with highest Borda score
        best_provider = max(borda_scores.items(), key=lambda x: x[1])[0]
        best_result = next(r for r in results if r.provider == best_provider)
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=normalized_scores[best_provider],
                explanation=f"Selected result from {best_provider} with Borda score {borda_scores[best_provider]:.2f}"
            ),
            provider_votes=normalized_scores
        )
    
    async def _evaluate_confidence(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> ConfidenceScore:
        """Evaluate confidence in a result using LLM."""
        prompt = f"""
        Evaluate the confidence in this {task_type} result:
        {result.result}
        
        Consider:
        1. Completeness of the response
        2. Logical consistency
        3. Relevance to the task
        4. Quality of the output
        
        Provide a confidence score (0.0 to 1.0) and explanation.
        """
        
        evaluation = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            model="gpt-4",
            **kwargs
        )
        
        # Parse confidence score from evaluation
        score = self._parse_confidence_score(evaluation.result)
        explanation = self._parse_confidence_explanation(evaluation.result)
        
        return ConfidenceScore(
            score=score,
            explanation=explanation,
            metadata={"raw_evaluation": evaluation.result}
        )
    
    async def _evaluate_with_llm(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> str:
        """Evaluate a result using LLM."""
        prompt = f"""
        Evaluate this {task_type} result:
        {result.result}
        
        Consider:
        1. Accuracy and correctness
        2. Completeness
        3. Clarity and coherence
        4. Relevance to the task
        
        Provide a detailed evaluation with a numerical score (0-100).
        """
        
        evaluation = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            model="gpt-4",
            **kwargs
        )
        
        return evaluation.result
    
    async def _get_provider_ranking(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> List[str]:
        """Get ranking of results from a provider."""
        prompt = f"""
        Rank the following {task_type} results from best to worst:
        {result.result}
        
        Consider:
        1. Quality and accuracy
        2. Completeness
        3. Relevance
        4. Clarity
        
        Provide a ranked list of provider names.
        """
        
        ranking = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            model="gpt-4",
            **kwargs
        )
        
        return self._parse_ranking(ranking.result)
    
    def _parse_confidence_score(self, evaluation: str) -> float:
        """Parse confidence score from evaluation text."""
        try:
            # Look for score in format "score: X" or "confidence: X"
            import re
            score_match = re.search(r'(?:score|confidence):\s*(\d*\.?\d+)', evaluation.lower())
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        except:
            pass
        return 0.5  # Default score if parsing fails
    
    def _parse_confidence_explanation(self, evaluation: str) -> str:
        """Parse confidence explanation from evaluation text."""
        try:
            # Look for explanation after "explanation:" or "reason:"
            import re
            explanation_match = re.search(r'(?:explanation|reason):\s*(.+?)(?:\n|$)', evaluation, re.IGNORECASE)
            if explanation_match:
                return explanation_match.group(1).strip()
        except:
            pass
        return "No explanation provided"
    
    def _parse_llm_evaluation(self, evaluation: str) -> float:
        """Parse numerical score from LLM evaluation."""
        try:
            # Look for score in format "score: X" or "rating: X"
            import re
            score_match = re.search(r'(?:score|rating):\s*(\d+)', evaluation.lower())
            if score_match:
                score = float(score_match.group(1))
                return min(max(score/100, 0.0), 1.0)  # Convert to 0-1 range
        except:
            pass
        return 0.5  # Default score if parsing fails
    
    def _parse_ranking(self, ranking_text: str) -> List[str]:
        """Parse provider ranking from text."""
        try:
            # Look for numbered list or comma-separated list
            import re
            providers = re.findall(r'\d+\.\s*(\w+)|,\s*(\w+)', ranking_text)
            return [p[0] or p[1] for p in providers if p[0] or p[1]]
        except:
            return []
    
    def _calculate_borda_score(self, ranking: List[str], total_providers: int) -> float:
        """Calculate Borda count score for a ranking."""
        if not ranking:
            return 0.0
        
        # Assign points based on position (highest for first place)
        points = {provider: total_providers - i for i, provider in enumerate(ranking)}
        
        # Normalize scores
        total_points = sum(points.values())
        return total_points / (total_providers * (total_providers + 1) / 2) 