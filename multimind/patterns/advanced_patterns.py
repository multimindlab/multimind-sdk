"""
Advanced RAG patterns including multi-hop retrieval, RAG-Fusion, Graph RAG, and self-improvement.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ..memory import TokenAwareMemory
from ..models.base import BaseLLM
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Any:
    """Extract and parse the first JSON object or array from model output."""
    match = re.search(r"\{.*\}|\[.*\]", str(text), re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {text!r}")
    return json.loads(match.group())


def _format_docs(docs: List[Dict[str, Any]], max_chars: int = 500) -> str:
    return "\n\n".join(
        f"[{i + 1}] {str(doc.get('content', ''))[:max_chars]}" for i, doc in enumerate(docs)
    )


@dataclass
class RetrievalStep:
    """Represents a step in multi-hop retrieval."""

    query: str
    retrieved_docs: List[Dict[str, Any]]
    reasoning: str
    confidence: float


@dataclass
class FusionResult:
    """Represents a result from RAG-Fusion."""

    query: str
    original_results: List[Dict[str, Any]]
    fused_results: List[Dict[str, Any]]
    fusion_scores: List[float]
    reasoning: str


class MultiHopRetriever:
    """Implements multi-hop retrieval with reasoning."""

    def __init__(
        self,
        model: BaseLLM,
        retriever: HybridRetriever,
        max_hops: int = 3,
        confidence_threshold: float = 0.7,
        **kwargs,
    ):
        self.model = model
        self.retriever = retriever
        self.max_hops = max_hops
        self.confidence_threshold = confidence_threshold
        self.kwargs = kwargs

    async def retrieve(
        self, query: str, initial_context: Optional[List[Dict[str, Any]]] = None, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[RetrievalStep]]:
        """
        Perform multi-hop retrieval with reasoning.

        Args:
            query: Initial query
            initial_context: Optional initial context
            **kwargs: Additional parameters

        Returns:
            Tuple of (retrieved documents, retrieval steps)
        """
        steps = []
        current_query = query
        retrieved_docs = initial_context or []
        seen_docs = set()

        for hop in range(self.max_hops):
            # Retrieve documents
            docs = await self.retriever.retrieve(
                query=current_query, documents=retrieved_docs, **kwargs
            )

            # Filter out seen documents
            new_docs = [doc for doc in docs if doc["id"] not in seen_docs]
            if not new_docs:
                break

            # Add to seen documents
            seen_docs.update(doc["id"] for doc in new_docs)
            retrieved_docs.extend(new_docs)

            # Generate reasoning and next query
            reasoning, next_query, confidence = await self._generate_reasoning(
                query=current_query, docs=new_docs, **kwargs
            )

            # Record step
            steps.append(
                RetrievalStep(
                    query=current_query,
                    retrieved_docs=new_docs,
                    reasoning=reasoning,
                    confidence=confidence,
                )
            )

            # Check if we should stop
            if confidence < self.confidence_threshold:
                break

            current_query = next_query

        return retrieved_docs, steps

    async def _generate_reasoning(
        self, query: str, docs: List[Dict[str, Any]], **kwargs
    ) -> Tuple[str, str, float]:
        """Generate reasoning and next query."""
        prompt = (
            "You are guiding a multi-hop retrieval process.\n"
            f"Current query: {query}\n"
            f"Retrieved documents:\n{_format_docs(docs)}\n\n"
            "Respond with only a JSON object with keys:\n"
            '"reasoning": why these documents do or do not answer the query,\n'
            '"next_query": a follow-up query to retrieve missing information,\n'
            '"confidence": a number between 0 and 1 for how fully the documents answer the query.'
        )
        response = await self.model.generate(prompt)
        try:
            parsed = _extract_json(response)
            reasoning = str(parsed["reasoning"])
            next_query = str(parsed["next_query"])
            confidence = max(0.0, min(1.0, float(parsed["confidence"])))
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"Could not parse reasoning from model output: {response!r}") from e
        return reasoning, next_query, confidence


class RAGFusion:
    """Implements RAG-Fusion for improved retrieval."""

    def __init__(self, model: BaseLLM, retriever: HybridRetriever, num_queries: int = 3, **kwargs):
        self.model = model
        self.retriever = retriever
        self.num_queries = num_queries
        self.kwargs = kwargs

    async def fuse(self, query: str, **kwargs) -> FusionResult:
        """
        Perform RAG-Fusion retrieval.

        Args:
            query: Original query
            **kwargs: Additional parameters

        Returns:
            Fusion result with original and fused results
        """
        # Generate query variations
        variations = await self._generate_query_variations(query, **kwargs)

        # Retrieve for each variation, keeping per-variation rankings
        rankings = []
        all_results = []
        for variation in variations:
            results = await self.retriever.retrieve(query=variation, **kwargs)
            rankings.append(results)
            all_results.extend(results)

        # Remove duplicates
        unique_results = self._remove_duplicates(all_results)

        # Calculate fusion scores
        fusion_scores = self._calculate_fusion_scores(rankings, unique_results)

        # Sort by fusion scores
        sorted_pairs = sorted(
            zip(fusion_scores, unique_results), key=lambda pair: pair[0], reverse=True
        )
        sorted_scores = [score for score, _ in sorted_pairs]
        sorted_results = [result for _, result in sorted_pairs]

        # Generate reasoning
        reasoning = await self._generate_fusion_reasoning(
            query=query, results=sorted_results, **kwargs
        )

        return FusionResult(
            query=query,
            original_results=all_results,
            fused_results=sorted_results,
            fusion_scores=sorted_scores,
            reasoning=reasoning,
        )

    async def _generate_query_variations(self, query: str, **kwargs) -> List[str]:
        """Generate query variations."""
        num_variations = self.num_queries - 1
        if num_variations <= 0:
            return [query]

        prompt = (
            f"Generate {num_variations} alternative phrasings of the following search query. "
            "Each variation should preserve the original intent but use different wording.\n"
            f"Query: {query}\n\n"
            "Respond with only the variations, one per line, without numbering."
        )
        response = await self.model.generate(prompt)
        variations = []
        for line in str(response).splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", line).strip()
            if cleaned and cleaned != query:
                variations.append(cleaned)
        if not variations:
            raise ValueError(f"Could not parse query variations from model output: {response!r}")
        return [query] + variations[:num_variations]

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results."""
        seen = set()
        unique = []
        for result in results:
            if result["id"] not in seen:
                seen.add(result["id"])
                unique.append(result)
        return unique

    def _calculate_fusion_scores(
        self,
        rankings: List[List[Dict[str, Any]]],
        unique_results: List[Dict[str, Any]],
        rrf_k: int = 60,
    ) -> List[float]:
        """Calculate reciprocal rank fusion scores across per-variation rankings."""
        scores: Dict[Any, float] = {}
        for ranking in rankings:
            for rank, result in enumerate(ranking):
                scores[result["id"]] = scores.get(result["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
        return [scores.get(result["id"], 0.0) for result in unique_results]

    async def _generate_fusion_reasoning(
        self, query: str, results: List[Dict[str, Any]], **kwargs
    ) -> str:
        """Generate reasoning about fusion results."""
        if not results:
            return "No results were retrieved for the query or its variations."
        prompt = (
            f"Query: {query}\n"
            f"Top fused retrieval results:\n{_format_docs(results[:5])}\n\n"
            "Briefly explain why these results are relevant to the query."
        )
        return str(await self.model.generate(prompt))


class GraphRAG:
    """Implements Graph RAG for structured knowledge retrieval."""

    def __init__(self, model: BaseLLM, retriever: HybridRetriever, **kwargs):
        self.model = model
        self.retriever = retriever
        self.graph = nx.DiGraph()
        self.kwargs = kwargs

    async def add_document(self, doc: Dict[str, Any], **kwargs) -> None:
        """
        Add document to knowledge graph.

        Args:
            doc: Document to add
            **kwargs: Additional parameters
        """
        # Extract entities and relationships
        entities, relationships = await self._extract_knowledge(doc, **kwargs)

        # Add to graph
        self.graph.add_node(
            doc["id"], type="document", content=doc["content"], metadata=doc.get("metadata", {})
        )

        for entity in entities:
            self.graph.add_node(entity["id"], type="entity", **entity)
            self.graph.add_edge(doc["id"], entity["id"], type="contains")

        for rel in relationships:
            self.graph.add_edge(
                rel["source"], rel["target"], type=rel["type"], **rel.get("metadata", {})
            )

    async def retrieve(
        self, query: str, **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve documents and entities from knowledge graph.

        Args:
            query: Query to retrieve for
            **kwargs: Additional parameters

        Returns:
            Tuple of (retrieved documents, retrieved entities)
        """
        # Extract query entities
        query_entities = await self._extract_entities(query, **kwargs)

        # Find relevant documents and entities
        relevant_docs = []
        relevant_entities = []

        for entity in query_entities:
            # Find documents containing entity
            docs = self._find_documents_with_entity(entity)
            relevant_docs.extend(docs)

            # Find related entities
            entities = self._find_related_entities(entity)
            relevant_entities.extend(entities)

        # Remove duplicates
        relevant_docs = self._remove_duplicates(relevant_docs)
        relevant_entities = self._remove_duplicates(relevant_entities)

        return relevant_docs, relevant_entities

    async def _extract_knowledge(
        self, doc: Dict[str, Any], **kwargs
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from document."""
        prompt = (
            "Extract entities and relationships from the following document.\n"
            f"Document:\n{str(doc.get('content', ''))[:2000]}\n\n"
            "Respond with only a JSON object of the form:\n"
            '{"entities": [{"id": "<unique-id>", "name": "<name>", "type": "<type>"}],\n'
            ' "relationships": [{"source": "<entity-id>", "target": "<entity-id>", '
            '"type": "<relation>"}]}'
        )
        response = await self.model.generate(prompt)
        try:
            parsed = _extract_json(response)
            entities = list(parsed["entities"])
            relationships = list(parsed["relationships"])
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Could not parse knowledge extraction from model output: {response!r}"
            ) from e
        return entities, relationships

    async def _extract_entities(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities from query."""
        prompt = (
            "Extract the entities mentioned in the following query.\n"
            f"Query: {query}\n\n"
            "Respond with only a JSON array of the form:\n"
            '[{"id": "<unique-id>", "name": "<name>", "type": "<type>"}]'
        )
        response = await self.model.generate(prompt)
        try:
            entities = list(_extract_json(response))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not parse entities from model output: {response!r}") from e
        return entities

    def _find_documents_with_entity(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find documents containing entity."""
        docs = []
        for _, doc_id in self.graph.edges(entity["id"]):
            if self.graph.nodes[doc_id]["type"] == "document":
                docs.append({"id": doc_id, **self.graph.nodes[doc_id]})
        return docs

    def _find_related_entities(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities related to given entity."""
        entities = []
        for _, target in self.graph.edges(entity["id"]):
            if self.graph.nodes[target]["type"] == "entity":
                entities.append({"id": target, **self.graph.nodes[target]})
        return entities

    def _remove_duplicates(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate items."""
        seen = set()
        unique = []
        for item in items:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)
        return unique


class SelfImprovingRAG:
    """Implements self-improving RAG with feedback loops."""

    def __init__(
        self,
        model: BaseLLM,
        retriever: HybridRetriever,
        memory: TokenAwareMemory,
        peft_tuner: Optional[Any] = None,
        retrain_threshold: float = 0.8,
        retrain_window: int = 10,
        retrain_cooldown: int = 3600,
        **kwargs,
    ):
        self.model = model
        self.retriever = retriever
        self.memory = memory
        self.peft_tuner = peft_tuner
        self.retrain_threshold = retrain_threshold
        self.retrain_window = retrain_window
        self.retrain_cooldown = retrain_cooldown
        self.kwargs = kwargs

        # Feedback tracking
        self.feedback_history: List[Dict[str, Any]] = []
        self.last_retrain_time: Optional[float] = None

    async def process_query(self, query: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process query with self-improvement.

        Args:
            query: Query to process
            **kwargs: Additional parameters

        Returns:
            Tuple of (response, metadata)
        """
        # Get relevant memory
        memory_items = await self.memory.get_relevant_memory(query=query, **kwargs)

        # Retrieve documents
        docs = await self.retriever.retrieve(query=query, **kwargs)

        # Generate response
        response, metadata = await self._generate_response(
            query=query, docs=docs, memory=memory_items, **kwargs
        )

        # Evaluate response
        evaluation = await self._evaluate_response(
            query=query, response=response, docs=docs, **kwargs
        )

        # Learn from feedback
        await self._learn_from_feedback(
            query=query, response=response, evaluation=evaluation, **kwargs
        )

        # Update memory
        await self.memory.add_conversation_turn(
            query=query,
            response=response,
            context=docs,
            metadata={"evaluation": evaluation, **metadata},
        )

        return response, metadata

    async def _generate_response(
        self, query: str, docs: List[Dict[str, Any]], memory: List[Dict[str, Any]], **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response with context."""
        sections = []
        if memory:
            sections.append(f"Conversation memory:\n{_format_docs(memory)}")
        if docs:
            sections.append(f"Retrieved documents:\n{_format_docs(docs)}")
        context = "\n\n".join(sections) if sections else "No context available."

        prompt = (
            "Answer the question using the provided context. "
            "If the context is insufficient, say so.\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )
        response = str(await self.model.generate(prompt))
        return response, {"num_docs": len(docs), "num_memory_items": len(memory)}

    async def _evaluate_response(
        self, query: str, response: str, docs: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Evaluate response quality."""
        prompt = (
            "Evaluate the following response.\n"
            f"Question: {query}\n"
            f"Context documents:\n{_format_docs(docs)}\n"
            f"Response: {response}\n\n"
            "Rate each criterion between 0 and 1 and respond with only a JSON object:\n"
            '{"relevance": <0-1>, "faithfulness": <0-1>, "coherence": <0-1>}'
        )
        raw = await self.model.generate(prompt)
        try:
            parsed = _extract_json(raw)
            return {
                key: max(0.0, min(1.0, float(parsed[key])))
                for key in ("relevance", "faithfulness", "coherence")
            }
        except (ValueError, KeyError, TypeError):
            # Unknown is reported as None rather than a fabricated score
            logger.warning("Could not parse evaluation scores from model output: %r", raw)
            return {"relevance": None, "faithfulness": None, "coherence": None}

    async def _learn_from_feedback(
        self, query: str, response: str, evaluation: Dict[str, Any], **kwargs
    ) -> None:
        """Learn from feedback to improve future responses."""
        # This is a placeholder implementation
        # In practice, you would:
        # 1. Update retrieval strategy
        # 2. Adjust prompt templates
        # 3. Fine-tune models
        # 4. Update memory importance
        pass

    def submit_feedback(self, query: str, response: str, feedback: Dict[str, Any]) -> None:
        """
        Submit feedback for a query-response pair.

        Args:
            query: The original query
            response: The generated response
            feedback: Feedback dictionary (e.g., {"thumbs": "down"})
        """
        feedback_entry = {
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": time.time(),
        }
        self.feedback_history.append(feedback_entry)

        # Check if retraining is needed
        if self.peft_tuner is not None:
            self._check_retrain_conditions()

    async def analyze_feedback(self) -> Dict[str, Any]:
        """
        Analyze collected feedback and return statistics.

        Returns:
            Dictionary with feedback analytics
        """
        if not self.feedback_history:
            return {
                "stats": {
                    "total_feedbacks": 0,
                    "positive": 0,
                    "negative": 0,
                    "average_quality": 0.0,
                }
            }

        total = len(self.feedback_history)
        positive = sum(
            1 for f in self.feedback_history if f.get("feedback", {}).get("thumbs") == "up"
        )
        negative = sum(
            1 for f in self.feedback_history if f.get("feedback", {}).get("thumbs") == "down"
        )

        # Calculate average quality (simple heuristic)
        quality_scores = []
        for f in self.feedback_history:
            thumbs = f.get("feedback", {}).get("thumbs", "")
            if thumbs == "up":
                quality_scores.append(1.0)
            elif thumbs == "down":
                quality_scores.append(0.0)
            else:
                # If no explicit feedback, assume neutral
                quality_scores.append(0.5)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "stats": {
                "total_feedbacks": total,
                "positive": positive,
                "negative": negative,
                "average_quality": avg_quality,
            }
        }

    def _check_retrain_conditions(self) -> None:
        """Check if retraining conditions are met and trigger retraining if needed."""
        if self.peft_tuner is None:
            return

        # Check cooldown period
        current_time = time.time()
        if self.last_retrain_time is not None:
            time_since_retrain = current_time - self.last_retrain_time
            if time_since_retrain < self.retrain_cooldown:
                return

        # Check if we have enough feedback in the window
        recent_feedback = self.feedback_history[-self.retrain_window :]
        if len(recent_feedback) < self.retrain_window:
            return

        # Calculate average quality for recent feedback
        quality_scores = []
        for f in recent_feedback:
            thumbs = f.get("feedback", {}).get("thumbs", "")
            if thumbs == "up":
                quality_scores.append(1.0)
            elif thumbs == "down":
                quality_scores.append(0.0)
            else:
                quality_scores.append(0.5)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Trigger retraining if quality is below threshold
        if avg_quality < self.retrain_threshold:
            logger.info(
                "[SelfImprovingRAG] Quality (%.2f) below threshold (%s), triggering retraining...",
                avg_quality,
                self.retrain_threshold,
            )
            self._trigger_retraining(recent_feedback)
            self.last_retrain_time = current_time

    def _trigger_retraining(self, training_data: List[Dict[str, Any]]) -> None:
        """Trigger model retraining with collected feedback data."""
        if self.peft_tuner is None:
            return

        # Prepare training data format expected by PEFT tuner
        train_data = []
        for entry in training_data:
            train_data.append(
                {
                    "query": entry["query"],
                    "response": entry["response"],
                    "feedback": entry["feedback"],
                }
            )

        # Train and save model
        self.peft_tuner.train(train_data)
        self.peft_tuner.save_model()
        logger.info("[SelfImprovingRAG] Retraining completed on %s samples.", len(train_data))
