"""
Enhanced retrieval system with hierarchical, temporal-aware, and domain-specific capabilities.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import asyncio
import numpy as np
from datetime import datetime
import networkx as nx
from ..models.base import BaseLLM
from .retrieval import HybridRetriever

@dataclass
class TemporalContext:
    """Represents temporal context for retrieval."""
    query_time: datetime
    document_time: datetime
    time_difference: float
    temporal_relevance: float
    temporal_metadata: Dict[str, Any]

@dataclass
class HierarchicalContext:
    """Represents hierarchical context for retrieval."""
    level: int
    parent_id: Optional[str]
    child_ids: List[str]
    path: List[str]
    hierarchy_score: float

@dataclass
class DomainContext:
    """Represents domain-specific context for retrieval."""
    domain: str
    domain_score: float
    domain_metadata: Dict[str, Any]
    domain_entities: List[Dict[str, Any]]

class RetrievalType(Enum):
    """Types of retrieval strategies."""
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    DOMAIN = "domain"
    MULTI_LINGUAL = "multi_lingual"
    HYBRID = "hybrid"

class EnhancedRetriever:
    """
    Enhanced retriever with advanced capabilities.

    Features:
    - Weighted, cross-strategy, and explainable result fusion
    - Per-strategy fusion weights (user-configurable or adaptive)
    - User feedback loop to adapt weights
    - Plugin system for custom fusion logic
    - Each result includes an explanation of its score
    Usage:
        retriever = EnhancedRetriever(...)
        results = await retriever.retrieve(...)
        # After user feedback:
        retriever.record_feedback(strategy='hierarchical', success=True, feedback=1.0)
    """

    def __init__(
        self,
        model: BaseLLM,
        base_retriever: HybridRetriever,
        fusion_weights: Optional[Dict[str, float]] = None,
        custom_fusion_fn: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize enhanced retriever.
        
        Args:
            model: Language model
            base_retriever: Base retriever
            fusion_weights: Per-strategy fusion weights
            custom_fusion_fn: Custom fusion function
            **kwargs: Additional parameters
        """
        self.model = model
        self.base_retriever = base_retriever
        self.hierarchy_graph = nx.DiGraph()
        self.domain_classifier = None  # Initialize domain classifier
        self.language_detector = None  # Initialize language detector
        self.kwargs = kwargs
        self.fusion_weights = fusion_weights or {
            "hierarchical": 1.0,
            "temporal": 1.0,
            "domain": 1.0,
            "multi_lingual": 1.0
        }
        self.feedback_history = {k: [] for k in self.fusion_weights}
        self.custom_fusion_fn = custom_fusion_fn

    async def retrieve(
        self,
        query: str,
        retrieval_type: RetrievalType = RetrievalType.HYBRID,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with enhanced capabilities.
        
        Args:
            query: Query to retrieve for
            retrieval_type: Type of retrieval to use
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        if retrieval_type == RetrievalType.HIERARCHICAL:
            return await self._hierarchical_retrieve(query, **kwargs)
        elif retrieval_type == RetrievalType.TEMPORAL:
            return await self._temporal_retrieve(query, **kwargs)
        elif retrieval_type == RetrievalType.DOMAIN:
            return await self._domain_retrieve(query, **kwargs)
        elif retrieval_type == RetrievalType.MULTI_LINGUAL:
            return await self._multi_lingual_retrieve(query, **kwargs)
        else:
            return await self._hybrid_retrieve(query, **kwargs)

    async def _hierarchical_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hierarchical retrieval."""
        # Get base results
        base_results = await self.base_retriever.retrieve(query, **kwargs)
        
        # Build hierarchy context
        hierarchy_contexts = []
        for doc in base_results:
            context = await self._get_hierarchical_context(doc, **kwargs)
            hierarchy_contexts.append(context)
        
        # Score based on hierarchy
        scored_results = []
        for doc, context in zip(base_results, hierarchy_contexts):
            # Calculate hierarchy score
            hierarchy_score = await self._calculate_hierarchy_score(
                query=query,
                context=context,
                **kwargs
            )
            
            # Update document score
            doc["score"] = doc.get("score", 0.0) * hierarchy_score
            doc["hierarchy_context"] = context
            
            scored_results.append(doc)
        
        # Sort by combined score
        return sorted(scored_results, key=lambda x: x["score"], reverse=True)

    async def _temporal_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform temporal-aware retrieval."""
        # Get base results
        base_results = await self.base_retriever.retrieve(query, **kwargs)
        
        # Get query temporal context
        query_time = kwargs.get("query_time", datetime.now())
        
        # Build temporal context
        temporal_contexts = []
        for doc in base_results:
            context = await self._get_temporal_context(
                query_time=query_time,
                document=doc,
                **kwargs
            )
            temporal_contexts.append(context)
        
        # Score based on temporal relevance
        scored_results = []
        for doc, context in zip(base_results, temporal_contexts):
            # Calculate temporal score
            temporal_score = await self._calculate_temporal_score(
                query=query,
                context=context,
                **kwargs
            )
            
            # Update document score
            doc["score"] = doc.get("score", 0.0) * temporal_score
            doc["temporal_context"] = context
            
            scored_results.append(doc)
        
        # Sort by combined score
        return sorted(scored_results, key=lambda x: x["score"], reverse=True)

    async def _domain_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform domain-specific retrieval."""
        # Get base results
        base_results = await self.base_retriever.retrieve(query, **kwargs)
        
        # Get query domain context
        query_domain = await self._detect_domain(query, **kwargs)
        
        # Build domain context
        domain_contexts = []
        for doc in base_results:
            context = await self._get_domain_context(
                query_domain=query_domain,
                document=doc,
                **kwargs
            )
            domain_contexts.append(context)
        
        # Score based on domain relevance
        scored_results = []
        for doc, context in zip(base_results, domain_contexts):
            # Calculate domain score
            domain_score = await self._calculate_domain_score(
                query=query,
                context=context,
                **kwargs
            )
            
            # Update document score
            doc["score"] = doc.get("score", 0.0) * domain_score
            doc["domain_context"] = context
            
            scored_results.append(doc)
        
        # Sort by combined score
        return sorted(scored_results, key=lambda x: x["score"], reverse=True)

    async def _multi_lingual_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform multi-lingual retrieval."""
        # Detect query language
        query_language = await self._detect_language(query, **kwargs)
        
        # Get base results
        base_results = await self.base_retriever.retrieve(query, **kwargs)
        
        # Process each document
        processed_results = []
        for doc in base_results:
            # Detect document language
            doc_language = await self._detect_language(
                doc["content"],
                **kwargs
            )
            
            # Translate if needed
            if doc_language != query_language:
                translated_content = await self._translate_content(
                    content=doc["content"],
                    source_lang=doc_language,
                    target_lang=query_language,
                    **kwargs
                )
                doc["translated_content"] = translated_content
            
            processed_results.append(doc)
        
        return processed_results

    async def _hybrid_retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining all strategies."""
        # Get results from each strategy
        hierarchical_results = await self._hierarchical_retrieve(
            query=query,
            **kwargs
        )
        temporal_results = await self._temporal_retrieve(
            query=query,
            **kwargs
        )
        domain_results = await self._domain_retrieve(
            query=query,
            **kwargs
        )
        multi_lingual_results = await self._multi_lingual_retrieve(
            query=query,
            **kwargs
        )
        
        # Combine results
        combined_results = self._combine_results(
            hierarchical_results=hierarchical_results,
            temporal_results=temporal_results,
            domain_results=domain_results,
            multi_lingual_results=multi_lingual_results,
            **kwargs
        )
        
        return combined_results

    async def _get_hierarchical_context(
        self,
        document: Dict[str, Any],
        **kwargs
    ) -> HierarchicalContext:
        """Get hierarchical context for document."""
        # This is a placeholder implementation
        return HierarchicalContext(
            level=0,
            parent_id=None,
            child_ids=[],
            path=[],
            hierarchy_score=0.0
        )

    async def _get_temporal_context(
        self,
        query_time: datetime,
        document: Dict[str, Any],
        **kwargs
    ) -> TemporalContext:
        """Get temporal context for document."""
        # Extract document time
        doc_time = self._extract_document_time(document)
        
        # Calculate time difference
        time_diff = abs((query_time - doc_time).total_seconds())
        
        return TemporalContext(
            query_time=query_time,
            document_time=doc_time,
            time_difference=time_diff,
            temporal_relevance=0.0,  # Calculate based on time difference
            temporal_metadata={}
        )

    async def _get_domain_context(
        self,
        query_domain: str,
        document: Dict[str, Any],
        **kwargs
    ) -> DomainContext:
        """Get domain context for document."""
        # Detect document domain
        doc_domain = await self._detect_domain(
            document["content"],
            **kwargs
        )
        
        # Extract domain entities
        entities = await self._extract_domain_entities(
            document["content"],
            domain=doc_domain,
            **kwargs
        )
        
        return DomainContext(
            domain=doc_domain,
            domain_score=0.0,  # Calculate based on domain match
            domain_metadata={},
            domain_entities=entities
        )

    async def _calculate_hierarchy_score(
        self,
        query: str,
        context: HierarchicalContext,
        **kwargs
    ) -> float:
        """Calculate hierarchy-based relevance score."""
        # Simple heuristic: higher level = lower score
        return max(0.0, 1.0 - 0.1 * context.level)

    async def _calculate_temporal_score(
        self,
        query: str,
        context: TemporalContext,
        **kwargs
    ) -> float:
        """Calculate temporal relevance score."""
        # Simple heuristic: more recent = higher score
        days_diff = context.time_difference / 86400  # seconds to days
        return max(0.0, 1.0 - 0.01 * days_diff)

    async def _calculate_domain_score(
        self,
        query: str,
        context: DomainContext,
        **kwargs
    ) -> float:
        """Calculate domain relevance score."""
        # Simple heuristic: exact domain match = 1.0, else 0.5
        if context.domain in query.lower():
            return 1.0
        return 0.5

    async def _detect_domain(
        self,
        text: str,
        **kwargs
    ) -> str:
        """Detect domain of text using keyword heuristics."""
        text_l = text.lower()
        if any(word in text_l for word in ["finance", "stock", "bank"]):
            return "finance"
        if any(word in text_l for word in ["health", "medical", "doctor"]):
            return "healthcare"
        if any(word in text_l for word in ["law", "legal", "court"]):
            return "legal"
        return "general"

    async def _detect_language(
        self,
        text: str,
        **kwargs
    ) -> str:
        """Detect language of text using simple heuristics."""
        if any(ord(c) > 128 for c in text):
            return "non-en"
        return "en"

    async def _translate_content(
        self,
        content: str,
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> str:
        """Translate content between languages (mock: append lang code)."""
        if source_lang == target_lang:
            return content
        return f"[Translated {source_lang}->{target_lang}]: {content}"

    async def _extract_domain_entities(
        self,
        text: str,
        domain: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Extract domain-specific entities using keyword matching."""
        entities = []
        if domain == "finance":
            for word in ["stock", "bank", "market"]:
                if word in text.lower():
                    entities.append({"entity": word, "type": "finance"})
        if domain == "healthcare":
            for word in ["doctor", "patient", "hospital"]:
                if word in text.lower():
                    entities.append({"entity": word, "type": "healthcare"})
        return entities

    def _extract_document_time(
        self,
        document: Dict[str, Any]
    ) -> datetime:
        """Extract time from document metadata or fallback to now."""
        if "timestamp" in document:
            try:
                return datetime.fromisoformat(document["timestamp"])
            except Exception:
                pass
        return datetime.now()

    def record_feedback(self, strategy: str, success: bool, feedback: float = None, ema_alpha: float = 0.2):
        """
        Record user or downstream feedback for a retrieval strategy.
        Updates fusion weights using exponential moving average (EMA).
        Args:
            strategy: Name of the retrieval strategy
            success: Whether the result was successful/correct
            feedback: Numeric feedback (e.g., user rating)
            ema_alpha: Smoothing factor for EMA (default 0.2)
        """
        if strategy not in self.fusion_weights:
            return
        # Use feedback if provided, else 1.0 for success, 0.0 for fail
        value = feedback if feedback is not None else (1.0 if success else 0.0)
        hist = self.feedback_history[strategy]
        if hist:
            prev = hist[-1]
            value = ema_alpha * value + (1 - ema_alpha) * prev
        hist.append(value)
        # Update fusion weight (normalize after all updates)
        self.fusion_weights[strategy] = value
        # Normalize weights
        total = sum(self.fusion_weights.values())
        for k in self.fusion_weights:
            self.fusion_weights[k] /= total if total > 0 else 1.0

    def set_fusion_weights(self, weights: Dict[str, float]):
        """Set fusion weights directly (overrides adaptive weights)."""
        self.fusion_weights = weights
        # Normalize
        total = sum(self.fusion_weights.values())
        for k in self.fusion_weights:
            self.fusion_weights[k] /= total if total > 0 else 1.0

    def set_custom_fusion(self, fn):
        """Set a custom fusion function (signature: (results_map, strategy_lists, strategy_names, **kwargs) -> List[Dict])."""
        self.custom_fusion_fn = fn

    def get_fusion_explanation(self) -> Dict[str, float]:
        """Return the current fusion weights for explainability."""
        return dict(self.fusion_weights)

    def _combine_results(
        self,
        hierarchical_results: List[Dict[str, Any]],
        temporal_results: List[Dict[str, Any]],
        domain_results: List[Dict[str, Any]],
        multi_lingual_results: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Combine results from different retrieval strategies using weighted, cross-strategy, and explainable fusion."""
        # Allow custom fusion function
        if self.custom_fusion_fn:
            return self.custom_fusion_fn(locals(), **kwargs)
        results_map = {}
        strategy_lists = [hierarchical_results, temporal_results, domain_results, multi_lingual_results]
        strategy_names = ["hierarchical", "temporal", "domain", "multi_lingual"]
        for strat_idx, results in enumerate(strategy_lists):
            for doc in results:
                doc_id = doc["id"]
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "id": doc_id,
                        "content": doc["content"],
                        "scores": [],
                        "confidences": [],
                        "contexts": {},
                        "strategies": set(),
                        "strategy_weights": []
                    }
                # Add score (weighted by fusion weight)
                strat = strategy_names[strat_idx]
                weight = self.fusion_weights.get(strat, 1.0)
                results_map[doc_id]["scores"].append(doc.get("score", 0.0) * weight)
                results_map[doc_id]["strategy_weights"].append(weight)
                # Add confidence if available
                conf = None
                if "confidence" in doc:
                    conf = doc["confidence"]
                elif "contexts" in doc and "confidence" in doc["contexts"]:
                    conf = doc["contexts"]["confidence"]
                if conf is not None:
                    results_map[doc_id]["confidences"].append(conf)
                # Add contexts
                for key in ["hierarchy_context", "temporal_context", "domain_context"]:
                    if key in doc:
                        results_map[doc_id]["contexts"][key] = doc[key]
                # Track which strategies this doc appeared in
                results_map[doc_id]["strategies"].add(strat)
        # Calculate combined scores
        combined_results = []
        for doc_id, data in results_map.items():
            # Weighted average if weights available, else mean
            if data["strategy_weights"]:
                weights = np.array(data["strategy_weights"])
                scores = np.array(data["scores"])
                combined_score = float(np.average(scores, weights=weights))
                explanation = f"Weighted average using fusion weights {weights.tolist()} from {len(data['scores'])} strategies."
            else:
                combined_score = float(np.mean(data["scores"]))
                explanation = f"Simple average from {len(data['scores'])} strategies."
            # Boost score if doc appears in multiple strategies
            n_strategies = len(data["strategies"])
            if n_strategies > 1:
                combined_score *= 1 + 0.1 * (n_strategies - 1)
                explanation += f" Boosted for appearing in {n_strategies} strategies."
            combined_results.append({
                "id": doc_id,
                "content": data["content"],
                "score": combined_score,
                "explanation": explanation,
                **data["contexts"]
            })
        # Sort by combined score
        return sorted(combined_results, key=lambda x: x["score"], reverse=True) 