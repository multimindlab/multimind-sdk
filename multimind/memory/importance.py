"""
Importance scoring implementation with hybrid approach.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from ..models.base import BaseLLM

class ImportanceScorer:
    """Hybrid importance scorer combining semantic relevance, recency, and task-specific importance."""

    def __init__(
        self,
        llm: BaseLLM,
        semantic_weight: float = 0.4,
        recency_weight: float = 0.3,
        task_weight: float = 0.3,
        recency_decay_hours: float = 24.0,
        min_confidence: float = 0.6,
        task_importance_threshold: float = 0.7,
        enable_learning: bool = True,
        learning_rate: float = 0.1
    ):
        """Initialize the importance scorer."""
        self.llm = llm
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.task_weight = task_weight
        self.recency_decay_hours = recency_decay_hours
        self.min_confidence = min_confidence
        self.task_importance_threshold = task_importance_threshold
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        
        # Performance tracking
        self.score_history: List[Dict[str, Any]] = []
        self.task_importance_history: Dict[str, List[float]] = {}
        self.weight_adjustments: List[Dict[str, Any]] = []

    async def score(
        self,
        content: str,
        metadata: Dict[str, Any],
        task_type: Optional[str] = None,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Calculate importance score using hybrid approach."""
        # Get individual scores
        semantic_score = await self._calculate_semantic_score(content, context)
        recency_score = self._calculate_recency_score(metadata.get("timestamp"))
        task_score = await self._calculate_task_score(content, task_type)
        
        # Calculate confidence for each component
        semantic_confidence = await self._calculate_confidence("semantic", content)
        recency_confidence = 1.0  # Recency is deterministic
        task_confidence = await self._calculate_confidence("task", content)
        
        # Adjust weights based on confidence
        adjusted_weights = self._adjust_weights_by_confidence(
            semantic_confidence,
            recency_confidence,
            task_confidence
        )
        
        # Calculate final score
        final_score = (
            adjusted_weights["semantic"] * semantic_score +
            adjusted_weights["recency"] * recency_score +
            adjusted_weights["task"] * task_score
        )
        
        # Record score history
        score_data = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "task_type": task_type,
            "scores": {
                "semantic": semantic_score,
                "recency": recency_score,
                "task": task_score,
                "final": final_score
            },
            "confidence": {
                "semantic": semantic_confidence,
                "recency": recency_confidence,
                "task": task_confidence
            },
            "weights": adjusted_weights
        }
        self.score_history.append(score_data)
        
        # Update weights if learning is enabled
        if self.enable_learning:
            await self._update_weights(score_data)
        
        return {
            "score": final_score,
            "components": {
                "semantic": semantic_score,
                "recency": recency_score,
                "task": task_score
            },
            "confidence": {
                "semantic": semantic_confidence,
                "recency": recency_confidence,
                "task": task_confidence
            },
            "weights": adjusted_weights
        }

    async def _calculate_semantic_score(
        self,
        content: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Calculate semantic relevance score."""
        try:
            if not context:
                return 0.5  # Default score if no context
            
            # Get content embedding
            content_embedding = await self.llm.embeddings(content)
            
            # Calculate similarity with context
            similarities = []
            for ctx in context:
                ctx_embedding = await self.llm.embeddings(ctx["content"])
                similarity = self._cosine_similarity(content_embedding, ctx_embedding)
                similarities.append(similarity)
            
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            print(f"Error calculating semantic score: {e}")
            return 0.5

    def _calculate_recency_score(self, timestamp: Optional[str]) -> float:
        """Calculate recency score."""
        if not timestamp:
            return 0.5
        
        try:
            content_time = datetime.fromisoformat(timestamp)
            age_hours = (datetime.now() - content_time).total_seconds() / 3600
            return float(np.exp(-age_hours / self.recency_decay_hours))
        except Exception as e:
            print(f"Error calculating recency score: {e}")
            return 0.5

    async def _calculate_task_score(
        self,
        content: str,
        task_type: Optional[str]
    ) -> float:
        """Calculate task-specific importance score."""
        if not task_type:
            return 0.5
        
        try:
            # Get historical importance for task type
            task_history = self.task_importance_history.get(task_type, [])
            if task_history:
                historical_importance = np.mean(task_history[-10:])  # Use last 10 scores
            else:
                historical_importance = 0.5
            
            # Analyze content relevance to task
            prompt = f"""
            Analyze how relevant this content is to the task type: {task_type}
            Content: {content}
            
            Return a relevance score between 0 and 1.
            """
            response = await self.llm.generate(prompt)
            try:
                relevance_score = float(response.strip())
            except ValueError:
                relevance_score = 0.5
            
            # Combine historical and current relevance
            task_score = 0.7 * relevance_score + 0.3 * historical_importance
            
            # Update task history
            if task_type not in self.task_importance_history:
                self.task_importance_history[task_type] = []
            self.task_importance_history[task_type].append(task_score)
            
            return task_score
            
        except Exception as e:
            print(f"Error calculating task score: {e}")
            return 0.5

    async def _calculate_confidence(
        self,
        component: str,
        content: str
    ) -> float:
        """Calculate confidence in the scoring component."""
        try:
            prompt = f"""
            Analyze the confidence in scoring this content for {component} importance.
            Content: {content}
            
            Consider:
            1. Content clarity and completeness
            2. Relevance to scoring criteria
            3. Potential ambiguity
            
            Return a confidence score between 0 and 1.
            """
            response = await self.llm.generate(prompt)
            try:
                confidence = float(response.strip())
            except ValueError:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5

    def _adjust_weights_by_confidence(
        self,
        semantic_confidence: float,
        recency_confidence: float,
        task_confidence: float
    ) -> Dict[str, float]:
        """Adjust weights based on confidence scores."""
        total_confidence = semantic_confidence + recency_confidence + task_confidence
        
        if total_confidence == 0:
            return {
                "semantic": self.semantic_weight,
                "recency": self.recency_weight,
                "task": self.task_weight
            }
        
        return {
            "semantic": (semantic_confidence / total_confidence) * self.semantic_weight,
            "recency": (recency_confidence / total_confidence) * self.recency_weight,
            "task": (task_confidence / total_confidence) * self.task_weight
        }

    async def _update_weights(self, score_data: Dict[str, Any]) -> None:
        """Update weights based on performance."""
        # Calculate performance metrics
        semantic_performance = score_data["scores"]["semantic"] * score_data["confidence"]["semantic"]
        recency_performance = score_data["scores"]["recency"] * score_data["confidence"]["recency"]
        task_performance = score_data["scores"]["task"] * score_data["confidence"]["task"]
        
        # Calculate weight adjustments
        total_performance = semantic_performance + recency_performance + task_performance
        if total_performance == 0:
            return
        
        semantic_adjustment = (semantic_performance / total_performance - self.semantic_weight) * self.learning_rate
        recency_adjustment = (recency_performance / total_performance - self.recency_weight) * self.learning_rate
        task_adjustment = (task_performance / total_performance - self.task_weight) * self.learning_rate
        
        # Apply adjustments
        self.semantic_weight = max(0.1, min(0.8, self.semantic_weight + semantic_adjustment))
        self.recency_weight = max(0.1, min(0.8, self.recency_weight + recency_adjustment))
        self.task_weight = max(0.1, min(0.8, self.task_weight + task_adjustment))
        
        # Normalize weights
        total = self.semantic_weight + self.recency_weight + self.task_weight
        self.semantic_weight /= total
        self.recency_weight /= total
        self.task_weight /= total
        
        # Record adjustment
        self.weight_adjustments.append({
            "timestamp": datetime.now().isoformat(),
            "old_weights": {
                "semantic": self.semantic_weight - semantic_adjustment,
                "recency": self.recency_weight - recency_adjustment,
                "task": self.task_weight - task_adjustment
            },
            "new_weights": {
                "semantic": self.semantic_weight,
                "recency": self.recency_weight,
                "task": self.task_weight
            },
            "performance": {
                "semantic": semantic_performance,
                "recency": recency_performance,
                "task": task_performance
            }
        })

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the scorer."""
        if not self.score_history:
            return {}
        
        recent_scores = self.score_history[-100:]  # Last 100 scores
        
        return {
            "average_scores": {
                "semantic": np.mean([s["scores"]["semantic"] for s in recent_scores]),
                "recency": np.mean([s["scores"]["recency"] for s in recent_scores]),
                "task": np.mean([s["scores"]["task"] for s in recent_scores]),
                "final": np.mean([s["scores"]["final"] for s in recent_scores])
            },
            "average_confidence": {
                "semantic": np.mean([s["confidence"]["semantic"] for s in recent_scores]),
                "recency": np.mean([s["confidence"]["recency"] for s in recent_scores]),
                "task": np.mean([s["confidence"]["task"] for s in recent_scores])
            },
            "current_weights": {
                "semantic": self.semantic_weight,
                "recency": self.recency_weight,
                "task": self.task_weight
            },
            "task_importance": {
                task: np.mean(scores[-10:])  # Average of last 10 scores
                for task, scores in self.task_importance_history.items()
            }
        } 