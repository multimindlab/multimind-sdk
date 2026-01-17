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
# Optional optuna import for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter tuning features will be disabled.")

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

class ProviderPerformanceTracker:
    """Tracks provider performance for adaptive weighting."""
    def __init__(self):
        self.metrics = {}
        # metrics: {provider: {"success": int, "fail": int, "latency": [float], "feedback": [float]}}

    def record(self, provider: str, success: bool, latency: float = None, feedback: float = None):
        if provider not in self.metrics:
            self.metrics[provider] = {"success": 0, "fail": 0, "latency": [], "feedback": []}
        if success:
            self.metrics[provider]["success"] += 1
        else:
            self.metrics[provider]["fail"] += 1
        if latency is not None:
            self.metrics[provider]["latency"].append(latency)
        if feedback is not None:
            self.metrics[provider]["feedback"].append(feedback)

    def get_weight(self, provider: str) -> float:
        m = self.metrics.get(provider, None)
        if not m:
            return 1.0
        # Weight: success rate * (1 / avg latency) * (avg feedback + 1)
        total = m["success"] + m["fail"]
        success_rate = m["success"] / total if total > 0 else 1.0
        avg_latency = np.mean(m["latency"]) if m["latency"] else 1.0
        avg_feedback = np.mean(m["feedback"]) if m["feedback"] else 0.0
        return success_rate * (1.0 / (avg_latency + 1e-3)) * (avg_feedback + 1.0)

    def get_all_weights(self, providers: List[str]) -> Dict[str, float]:
        return {p: self.get_weight(p) for p in providers}

    def submit_feedback(self, provider: str, feedback: float):
        self.record(provider, success=True, feedback=feedback)

class AdvancedEnsemble:
    """
    Advanced ensemble system for combining results from multiple providers.

    Features:
    - Weighted, confidence, semantic, and rank-based voting
    - Adaptive/learning-based weights using ProviderPerformanceTracker
    - Feedback loop: call record_outcome after each ensemble result to update weights
    - Custom ensemble strategies via plugin
    - Optuna-based hyperparameter tuning for ensemble weights (see tune_weights_with_optuna)
    """
    
    def __init__(self, router: Router):
        """Initialize the ensemble system."""
        self.router = router
        self.performance_tracker = ProviderPerformanceTracker()
        self.custom_strategies = {}  # name -> async function
    
    def register_strategy(self, name: str, strategy_fn):
        """Register a custom ensemble strategy. The function must be async and accept (results, task_type, **kwargs)."""
        self.custom_strategies[name] = strategy_fn

    async def combine_results(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        method: Union[EnsembleMethod, str],
        task_type: TaskType,
        **kwargs
    ) -> EnsembleResult:
        """
        Combine results using the specified method or a registered custom strategy.
        
        System behavior:
        - 2+ LLMs: Full ensemble logic
        - 1 LLM: Acts like fallback router (returns the single result)
        - 0 LLMs: Hard failure (raises exception)
        """
        # Filter out None results (failed providers)
        valid_results = [r for r in results if r is not None]
        
        # System behavior based on LLM count
        if len(valid_results) == 0:
            raise ValueError("No valid results provided. All providers failed. Hard failure.")
        elif len(valid_results) == 1:
            # Single LLM: Act like fallback router - just return the result
            single_result = valid_results[0]
            provider_name = self._get_provider_name(single_result)
            return EnsembleResult(
                result=single_result,
                confidence=ConfidenceScore(
                    score=1.0,
                    explanation=f"Single provider result from {provider_name} (fallback router mode)"
                ),
                provider_votes={provider_name: 1.0}
            )
        else:
            # 2+ LLMs: Full ensemble logic
            if isinstance(method, str) and method in self.custom_strategies:
                return await self.custom_strategies[method](valid_results, task_type, **kwargs)
            if method == EnsembleMethod.WEIGHTED_VOTING:
                return await self._weighted_voting(valid_results, **kwargs)
            elif method == EnsembleMethod.CONFIDENCE_CASCADE:
                return await self._confidence_cascade(valid_results, task_type, **kwargs)
            elif method == EnsembleMethod.PARALLEL_VOTING:
                return await self._parallel_voting(valid_results, task_type, **kwargs)
            elif method == EnsembleMethod.MAJORITY_VOTING:
                return await self._majority_voting(valid_results, **kwargs)
            elif method == EnsembleMethod.RANK_BASED:
                return await self._rank_based(valid_results, task_type, **kwargs)
            else:
                raise ValueError(f"Unsupported ensemble method: {method}")
    
    async def _weighted_voting(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        weights: Optional[Dict[str, float]] = None,
        use_adaptive_weights: bool = True,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using weighted voting (adaptive if enabled)."""
        if use_adaptive_weights or not weights:
            providers = [self._get_provider_name(result) for result in results]
            weights = self.performance_tracker.get_all_weights(providers)
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0 or not weights:
            # Fallback to equal weights if total is zero or weights is empty
            providers = [self._get_provider_name(result) for result in results]
            normalized_weights = {p: 1.0/len(providers) for p in providers} if providers else {}
        else:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
        # Calculate weighted scores for each result
        weighted_scores = []
        for result in results:
            provider_name = self._get_provider_name(result)
            weight = normalized_weights.get(provider_name, 0.0)
            weighted_scores.append((result, weight))
        # Select result with highest weight
        best_result, best_weight = max(weighted_scores, key=lambda x: x[1])
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=best_weight,
                explanation=f"Selected result from {self._get_provider_name(best_result)} with adaptive weight {best_weight:.2f}"
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
                    provider_votes={self._get_provider_name(r): c.score for r, c in confidence_scores}
                )
        
        # If no result meets threshold, return highest confidence
        best_result, best_confidence = confidence_scores[0]
        return EnsembleResult(
            result=best_result,
            confidence=best_confidence,
            provider_votes={self._get_provider_name(r): c.score for r, c in confidence_scores}
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
        normalized_scores = {self._get_provider_name(r): s/total_score for r, s in scores}
        
        # Select best result
        best_result, best_score = max(scores, key=lambda x: x[1])
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=best_score,
                explanation=f"Selected result from {self._get_provider_name(best_result)} with LLM evaluation score {best_score:.2f}"
            ),
            provider_votes=normalized_scores
        )
    
    async def _majority_voting(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        embedder=None,
        similarity_threshold: float = 0.8,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using semantic majority voting (embedding-based)."""
        # Use a default embedder if not provided
        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to string equality if no embedder available
                embedder = None

        texts = [self._extract_result_content(r) for r in results]
        if embedder is not None:
            embeddings = embedder.encode(texts, convert_to_tensor=True)
            import torch
            groups = []
            used = set()
            for i, emb in enumerate(embeddings):
                if i in used:
                    continue
                group = [i]
                for j in range(i+1, len(embeddings)):
                    if j in used:
                        continue
                    sim = torch.nn.functional.cosine_similarity(emb, embeddings[j], dim=0, eps=1e-6).item()
                    if sim >= similarity_threshold:
                        group.append(j)
                        used.add(j)
                used.add(i)
                groups.append(group)
            # Find largest group
            largest_group = max(groups, key=len)
            # Pick result with highest confidence/score in group
            group_results = [results[idx] for idx in largest_group]
            # Use score if available, else default to first
            def get_score(r):
                return getattr(r, 'score', 1.0) or 1.0
            best_result = max(group_results, key=get_score)
            vote_count = len(largest_group)
            total_votes = len(results)
            explanation = f"Selected result by semantic majority voting: {vote_count}/{total_votes} semantically similar."
            provider_votes = {self._get_provider_name(r): 1.0 if idx in largest_group else 0.0 for idx, r in enumerate(results)}
        else:
            # Fallback: string equality
            result_counts = {}
            for result in results:
                key = self._extract_result_content(result)
                if key not in result_counts:
                    result_counts[key] = (result, 0)
                result_counts[key] = (result, result_counts[key][1] + 1)
            best_result, vote_count = max(result_counts.values(), key=lambda x: x[1])
            total_votes = len(results)
            explanation = f"Selected result with {vote_count}/{total_votes} votes (string equality fallback)"
            provider_votes = {self._get_provider_name(r): 1.0 for r in results}
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=vote_count/total_votes,
                explanation=explanation
            ),
            provider_votes=provider_votes
        )
    
    async def _rank_based(
        self,
        results: List[Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]],
        task_type: TaskType,
        **kwargs
    ) -> EnsembleResult:
        """Combine results using rank-based selection."""
        if not results:
            raise ValueError("Cannot perform rank-based selection on empty results list")
        
        # Get rankings from each provider
        rankings = await asyncio.gather(*[
            self._get_provider_ranking(result, task_type, **kwargs)
            for result in results
        ])
        
        # Calculate Borda count
        borda_scores = {}
        for result, ranking in zip(results, rankings):
            score = self._calculate_borda_score(ranking, len(results))
            borda_scores[self._get_provider_name(result)] = score
        
        # Normalize scores
        total_score = sum(borda_scores.values())
        if total_score == 0 or not borda_scores:
            # Fallback to equal scores if all scores are zero (e.g., all rankings failed)
            normalized_scores = {k: 1.0/len(borda_scores) for k in borda_scores.keys()} if borda_scores else {}
            # If no scores, just pick first result
            if not borda_scores:
                best_result = results[0] if results else None
                best_provider = self._get_provider_name(best_result) if best_result else "unknown"
            else:
                best_provider = max(borda_scores.items(), key=lambda x: x[1])[0]
                best_result = next(r for r in results if self._get_provider_name(r) == best_provider)
        else:
            normalized_scores = {k: v/total_score for k, v in borda_scores.items()}
            # Select result with highest Borda score
            best_provider = max(borda_scores.items(), key=lambda x: x[1])[0]
            best_result = next(r for r in results if self._get_provider_name(r) == best_provider)
        
        return EnsembleResult(
            result=best_result,
            confidence=ConfidenceScore(
                score=normalized_scores[best_provider],
                explanation=f"Selected result from {best_provider} with Borda score {borda_scores[best_provider]:.2f}"
            ),
            provider_votes=normalized_scores
        )
    
    def _extract_result_content(self, result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]) -> str:
        """Extract text content from any result type for evaluation."""
        if isinstance(result, GenerationResult):
            return result.text
        elif isinstance(result, EmbeddingResult):
            return f"Embedding vector of length {len(result.embedding)}"
        elif isinstance(result, ImageAnalysisResult):
            # Combine text, captions, and objects for evaluation
            parts = []
            if result.text:
                parts.append(f"Text: {result.text}")
            if result.captions:
                parts.append(f"Captions: {', '.join(result.captions)}")
            if result.objects:
                parts.append(f"Objects: {len(result.objects)} detected")
            return " | ".join(parts) if parts else "No content extracted"
        else:
            return str(result)
    
    def _get_provider_name(self, result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult]) -> str:
        """Extract provider name from any result type."""
        return getattr(result, 'provider', None) or getattr(result, 'provider_name', 'unknown')
    
    async def _evaluate_confidence(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> ConfidenceScore:
        """Evaluate confidence in a result using LLM."""
        content = self._extract_result_content(result)
        prompt = f"""
        Evaluate the confidence in this {task_type} result:
        {content}
        
        Consider:
        1. Completeness of the response
        2. Logical consistency
        3. Relevance to the task
        4. Quality of the output
        
        Provide a confidence score (0.0 to 1.0) and explanation.
        """
        
        provider_name = self._get_provider_name(result)
        evaluation_models = kwargs.get("evaluation_models", {})
        evaluation_providers = kwargs.get("evaluation_providers", {})
        eval_provider = evaluation_providers.get(provider_name, kwargs.get("evaluation_provider", provider_name))
        
        # Use provider-appropriate default model if not specified
        if provider_name in evaluation_models:
            eval_model = evaluation_models[provider_name]
        else:
            # Prefer OpenAI for evaluation if available, otherwise use provider-appropriate model
            if "openai" in self.router.providers:
                eval_provider = "openai"
                eval_model = kwargs.get("evaluation_model", "gpt-4")
            elif eval_provider == "ollama":
                eval_model = kwargs.get("evaluation_model", "mistral")
            elif eval_provider == "anthropic":
                eval_model = kwargs.get("evaluation_model", "claude-3-sonnet")
            else:
                eval_model = kwargs.get("evaluation_model", "gpt-4")
        
        route_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"evaluation_models", "evaluation_providers", "evaluation_model", "evaluation_provider"}
        }
        evaluation = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider=eval_provider,
            model=eval_model,
            **route_kwargs
        )
        
        # Parse confidence score from evaluation
        eval_content = self._extract_result_content(evaluation)
        score = self._parse_confidence_score(eval_content)
        explanation = self._parse_confidence_explanation(eval_content)
        
        return ConfidenceScore(
            score=score,
            explanation=explanation,
            metadata={"raw_evaluation": eval_content}
        )
    
    async def _evaluate_with_llm(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> str:
        """Evaluate a result using LLM."""
        content = self._extract_result_content(result)
        prompt = f"""
        Evaluate this {task_type} result:
        {content}
        
        Consider:
        1. Accuracy and correctness
        2. Completeness
        3. Clarity and coherence
        4. Relevance to the task
        
        Provide a detailed evaluation with a numerical score (0-100).
        """
        
        provider_name = self._get_provider_name(result)
        evaluation_models = kwargs.get("evaluation_models", {})
        evaluation_providers = kwargs.get("evaluation_providers", {})
        eval_provider = evaluation_providers.get(provider_name, kwargs.get("evaluation_provider", provider_name))
        
        # Use provider-appropriate default model if not specified
        if provider_name in evaluation_models:
            eval_model = evaluation_models[provider_name]
        else:
            # Prefer OpenAI for evaluation if available, otherwise use provider-appropriate model
            if "openai" in self.router.providers:
                eval_provider = "openai"
                eval_model = kwargs.get("evaluation_model", "gpt-4")
            elif eval_provider == "ollama":
                eval_model = kwargs.get("evaluation_model", "mistral")
            elif eval_provider == "anthropic":
                eval_model = kwargs.get("evaluation_model", "claude-3-sonnet")
            else:
                eval_model = kwargs.get("evaluation_model", "gpt-4")
        
        route_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"evaluation_models", "evaluation_providers", "evaluation_model", "evaluation_provider"}
        }
        evaluation = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider=eval_provider,
            model=eval_model,
            **route_kwargs
        )
        
        return self._extract_result_content(evaluation)
    
    async def _get_provider_ranking(
        self,
        result: Union[GenerationResult, EmbeddingResult, ImageAnalysisResult],
        task_type: TaskType,
        **kwargs
    ) -> List[str]:
        """Get ranking of results from a provider."""
        content = self._extract_result_content(result)
        prompt = f"""
        Rank the following {task_type} results from best to worst:
        {content}
        
        Consider:
        1. Quality and accuracy
        2. Completeness
        3. Relevance
        4. Clarity
        
        Provide a ranked list of provider names.
        """
        
        provider_name = self._get_provider_name(result)
        evaluation_models = kwargs.get("evaluation_models", {})
        evaluation_providers = kwargs.get("evaluation_providers", {})
        default_model = kwargs.get("evaluation_model", "gpt-4")
        
        # Use provider-specific model if available, otherwise use smart defaults
        eval_model = evaluation_models.get(provider_name)
        if not eval_model:
            # Use provider-appropriate default models
            if provider_name == "ollama":
                eval_model = "mistral"  # Ollama doesn't have gpt-4
            elif provider_name == "anthropic" or provider_name == "claude":
                eval_model = "claude-3-sonnet"
            else:
                eval_model = default_model  # Use gpt-4 for OpenAI and others
        
        eval_provider = evaluation_providers.get(provider_name, kwargs.get("evaluation_provider", provider_name))
        route_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"evaluation_models", "evaluation_providers", "evaluation_model", "evaluation_provider"}
        }
        ranking = await self.router.route(
            TaskType.TEXT_GENERATION,
            prompt,
            provider=eval_provider,
            model=eval_model,
            **route_kwargs
        )
        
        return self._parse_ranking(self._extract_result_content(ranking))
    
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
        """Calculate Borda count score for a ranking.
        
        Borda count: For a ranking of n items, the first place gets (n-1) points,
        second place gets (n-2) points, etc. The score is the sum of points
        for the provider's position in this ranking.
        """
        if not ranking or total_providers == 0:
            return 0.0
        
        # Calculate score based on position in ranking
        # First place (index 0) gets (total_providers - 1) points
        # Second place (index 1) gets (total_providers - 2) points, etc.
        score = 0.0
        for i, provider in enumerate(ranking):
            if i < total_providers:
                score += (total_providers - i - 1)
        
        return score

    def submit_feedback(self, provider: str, feedback: float):
        """Submit user feedback for a provider (1.0=good, 0.0=bad, or any float)."""
        self.performance_tracker.submit_feedback(provider, feedback)

    def record_outcome(self, provider: str, success: bool, latency: float = None, feedback: float = None, ema_alpha: float = 0.2):
        """
        Record the outcome of an ensemble decision for a provider.
        Updates the ProviderPerformanceTracker and adapts weights.
        Optionally uses exponential moving average (EMA) for latency/feedback.
        Args:
            provider: Provider name
            success: Whether the result was successful/correct
            latency: Latency of the provider's response
            feedback: User or downstream feedback (numeric)
            ema_alpha: Smoothing factor for EMA (default 0.2)
        """
        # If using EMA, update latency/feedback with smoothing
        m = self.performance_tracker.metrics.get(provider, None)
        if m and latency is not None:
            if m["latency"]:
                prev = m["latency"][-1]
                latency = ema_alpha * latency + (1 - ema_alpha) * prev
        if m and feedback is not None:
            if m["feedback"]:
                prev = m["feedback"][-1]
                feedback = ema_alpha * feedback + (1 - ema_alpha) * prev
        self.performance_tracker.record(provider, success, latency, feedback)

    def tune_weights_with_optuna(self, results, task_type, eval_fn, n_trials=30):
        """
        Tune ensemble weights using Optuna.
        Args:
            results: List of results (GenerationResult, etc.)
            task_type: TaskType
            eval_fn: Function to evaluate ensemble result (signature: (EnsembleResult) -> float, higher is better)
            n_trials: Number of Optuna trials
        Returns:
            Dict of best weights
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Please install optuna.")
        
        providers = [self._get_provider_name(r) for r in results]
        def objective(trial):
            weights = {p: trial.suggest_float(f"weight_{p}", 0.01, 1.0) for p in providers}
            # Normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            # Run weighted voting
            loop = asyncio.get_event_loop()
            ensemble_result = loop.run_until_complete(self._weighted_voting(results, weights=weights, use_adaptive_weights=False))
            score = eval_fn(ensemble_result)
            return score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_weights = {k: v for k, v in study.best_params.items() if k.startswith("weight_")}
        # Normalize
        total = sum(best_weights.values())
        best_weights = {k.replace("weight_", ""): v/total for k, v in best_weights.items()}
        return best_weights

    # In class docstring, add:
    """
    Usage:
        result = await ensemble.combine_results(...)
        # After user feedback or downstream evaluation:
        ensemble.record_outcome(result.result.provider, success=True, latency=..., feedback=...)
    """ 