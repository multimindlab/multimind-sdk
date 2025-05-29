"""
Contextual memory implementation that maintains conversation context and relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class ContextualMemory(BaseMemory):
    """Memory that maintains conversation context and relationships."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        context_window: int = 10,
        max_contexts: int = 100,
        context_similarity_threshold: float = 0.7,
        relationship_types: Optional[List[str]] = None,
        context_merge_threshold: float = 0.8,
        temporal_weight: float = 0.3,
        semantic_weight: float = 0.4,
        relationship_weight: float = 0.3,
        enable_summarization: bool = True,
        summarization_interval: int = 3600,  # 1 hour
        evolution_tracking: bool = True,
        min_evolution_confidence: float = 0.6
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.context_window = context_window
        self.max_contexts = max_contexts
        self.context_similarity_threshold = context_similarity_threshold
        self.relationship_types = relationship_types or [
            "follows",
            "references",
            "contradicts",
            "elaborates",
            "summarizes",
            "questions",
            "answers"
        ]
        self.context_merge_threshold = context_merge_threshold
        self.temporal_weight = temporal_weight
        self.semantic_weight = semantic_weight
        self.relationship_weight = relationship_weight
        self.enable_summarization = enable_summarization
        self.summarization_interval = summarization_interval
        self.evolution_tracking = evolution_tracking
        self.min_evolution_confidence = min_evolution_confidence
        
        # Initialize context storage
        self.contexts: List[Dict[str, Any]] = []
        self.context_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Set[str]] = {}  # context_id -> set of related context_ids
        self.context_weights: Dict[str, float] = {}  # context_id -> weight
        self.context_metadata: Dict[str, Dict[str, Any]] = {}  # context_id -> metadata
        self.context_summaries: Dict[str, str] = {}  # context_id -> summary
        self.context_evolution: Dict[str, List[Dict[str, Any]]] = {}  # context_id -> evolution history
        self.last_summarization = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message to the appropriate context."""
        # Create new context or merge with existing
        context_id = f"ctx_{len(self.contexts)}"
        new_context = {
            "id": context_id,
            "messages": [message],
            "timestamp": datetime.now().isoformat(),
            "relationships": set(),
            "metadata": {
                "topic": None,
                "sentiment": None,
                "entities": set(),
                "keywords": set(),
                "evolution_stage": "initial",
                "confidence": 1.0
            }
        }
        
        # Analyze message for context
        await self._analyze_context(new_context)
        
        # Find similar contexts
        similar_contexts = await self._find_similar_contexts(new_context)
        
        if similar_contexts:
            # Merge with most similar context
            most_similar = similar_contexts[0]
            if most_similar["similarity"] >= self.context_merge_threshold:
                await self._merge_contexts(most_similar["context_id"], new_context)
            else:
                # Create new context with relationships
                await self._create_context(new_context, similar_contexts)
        else:
            # Create new context
            await self._create_context(new_context)
        
        # Update context weights
        await self._update_context_weights()
        
        # Check for summarization
        if self.enable_summarization:
            current_time = datetime.now()
            if (current_time - self.last_summarization).total_seconds() > self.summarization_interval:
                await self._summarize_contexts()
        
        # Track context evolution
        if self.evolution_tracking:
            await self._track_context_evolution(context_id)
        
        # Maintain context window
        await self._maintain_context_window()
        
        await self.save()

    async def _analyze_context(self, context: Dict[str, Any]) -> None:
        """Analyze context for metadata and relationships."""
        try:
            # Analyze topic and sentiment
            prompt = f"""
            Analyze the following conversation context and determine:
            1. Main topic
            2. Overall sentiment
            3. Key entities
            4. Important keywords
            5. Context confidence (0-1)
            
            Context: {context['messages']}
            
            Return in format:
            Topic: <topic>
            Sentiment: <sentiment>
            Entities: <comma-separated entities>
            Keywords: <comma-separated keywords>
            Confidence: <confidence score>
            """
            response = await self.llm.generate(prompt)
            
            # Parse response
            lines = response.split('\n')
            for line in lines:
                if line.startswith('Topic:'):
                    context['metadata']['topic'] = line.split(':', 1)[1].strip()
                elif line.startswith('Sentiment:'):
                    context['metadata']['sentiment'] = line.split(':', 1)[1].strip()
                elif line.startswith('Entities:'):
                    entities = line.split(':', 1)[1].strip().split(',')
                    context['metadata']['entities'] = {e.strip() for e in entities}
                elif line.startswith('Keywords:'):
                    keywords = line.split(':', 1)[1].strip().split(',')
                    context['metadata']['keywords'] = {k.strip() for k in keywords}
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    context['metadata']['confidence'] = confidence
            
            # Get context embedding
            context_text = ' '.join(msg['content'] for msg in context['messages'])
            embedding = await self.llm.embeddings(context_text)
            self.context_embeddings.append(embedding)
            
        except Exception as e:
            print(f"Error analyzing context: {e}")

    async def _summarize_contexts(self) -> None:
        """Summarize contexts to maintain concise representation."""
        for context in self.contexts:
            if len(context['messages']) > self.context_window:
                try:
                    # Generate summary
                    prompt = f"""
                    Summarize the following conversation context while preserving key information:
                    
                    Context: {context['messages']}
                    
                    Return a concise summary that captures the main points and relationships.
                    """
                    summary = await self.llm.generate(prompt)
                    
                    # Update context
                    self.context_summaries[context['id']] = summary
                    
                    # Keep only recent messages
                    context['messages'] = context['messages'][-self.context_window:]
                    
                except Exception as e:
                    print(f"Error summarizing context: {e}")
        
        self.last_summarization = datetime.now()

    async def _track_context_evolution(self, context_id: str) -> None:
        """Track the evolution of a context over time."""
        if context_id not in self.context_evolution:
            self.context_evolution[context_id] = []
        
        context = next(ctx for ctx in self.contexts if ctx['id'] == context_id)
        
        try:
            # Analyze evolution
            prompt = f"""
            Analyze how this context has evolved and determine:
            1. Current stage of evolution
            2. Key changes or developments
            3. Confidence in evolution analysis (0-1)
            
            Context: {context['messages']}
            Previous evolution: {self.context_evolution[context_id]}
            
            Return in format:
            Stage: <evolution stage>
            Changes: <key changes>
            Confidence: <confidence score>
            """
            response = await self.llm.generate(prompt)
            
            # Parse response
            lines = response.split('\n')
            evolution_data = {
                "timestamp": datetime.now().isoformat(),
                "stage": None,
                "changes": None,
                "confidence": None
            }
            
            for line in lines:
                if line.startswith('Stage:'):
                    evolution_data['stage'] = line.split(':', 1)[1].strip()
                elif line.startswith('Changes:'):
                    evolution_data['changes'] = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    evolution_data['confidence'] = confidence
            
            if evolution_data['confidence'] >= self.min_evolution_confidence:
                self.context_evolution[context_id].append(evolution_data)
                context['metadata']['evolution_stage'] = evolution_data['stage']
            
        except Exception as e:
            print(f"Error tracking context evolution: {e}")

    async def get_context_summary(self, context_id: str) -> Optional[str]:
        """Get the summary of a specific context."""
        return self.context_summaries.get(context_id)

    async def get_context_evolution(self, context_id: str) -> List[Dict[str, Any]]:
        """Get the evolution history of a specific context."""
        return self.context_evolution.get(context_id, [])

    async def _find_similar_contexts(
        self,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find contexts similar to the given context."""
        if not self.contexts:
            return []
        
        # Get context embedding
        context_text = ' '.join(msg['content'] for msg in context['messages'])
        context_embedding = await self.llm.embeddings(context_text)
        
        # Calculate similarities
        similarities = []
        for i, existing_embedding in enumerate(self.context_embeddings):
            similarity = self._cosine_similarity(context_embedding, existing_embedding)
            if similarity >= self.context_similarity_threshold:
                similarities.append({
                    "context_id": self.contexts[i]["id"],
                    "similarity": similarity
                })
        
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    async def _create_context(
        self,
        context: Dict[str, Any],
        similar_contexts: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Create a new context with optional relationships."""
        # Add context
        self.contexts.append(context)
        self.context_metadata[context["id"]] = context["metadata"]
        self.context_weights[context["id"]] = 1.0
        
        # Add relationships
        if similar_contexts:
            for similar in similar_contexts:
                if similar["similarity"] >= self.context_similarity_threshold:
                    await self._add_relationship(context["id"], similar["context_id"])

    async def _merge_contexts(
        self,
        target_id: str,
        source_context: Dict[str, Any]
    ) -> None:
        """Merge source context into target context."""
        target_idx = next(
            i for i, ctx in enumerate(self.contexts)
            if ctx["id"] == target_id
        )
        
        # Merge messages
        self.contexts[target_idx]["messages"].extend(source_context["messages"])
        
        # Merge metadata
        target_metadata = self.context_metadata[target_id]
        source_metadata = source_context["metadata"]
        
        target_metadata["entities"].update(source_metadata["entities"])
        target_metadata["keywords"].update(source_metadata["keywords"])
        
        # Update embedding
        context_text = ' '.join(
            msg['content'] for msg in self.contexts[target_idx]["messages"]
        )
        new_embedding = await self.llm.embeddings(context_text)
        self.context_embeddings[target_idx] = new_embedding

    async def _add_relationship(
        self,
        context_id1: str,
        context_id2: str,
        relationship_type: Optional[str] = None
    ) -> None:
        """Add relationship between two contexts."""
        if relationship_type is None:
            # Determine relationship type
            try:
                prompt = f"""
                Determine the relationship type between these contexts:
                Context 1: {self.contexts[0]['messages']}
                Context 2: {self.contexts[1]['messages']}
                
                Choose from: {', '.join(self.relationship_types)}
                """
                response = await self.llm.generate(prompt)
                relationship_type = response.strip()
                
                if relationship_type not in self.relationship_types:
                    relationship_type = "follows"
            except Exception as e:
                print(f"Error determining relationship type: {e}")
                relationship_type = "follows"
        
        # Add bidirectional relationship
        if context_id1 not in self.relationships:
            self.relationships[context_id1] = set()
        if context_id2 not in self.relationships:
            self.relationships[context_id2] = set()
        
        self.relationships[context_id1].add(f"{context_id2}:{relationship_type}")
        self.relationships[context_id2].add(f"{context_id1}:{relationship_type}")

    async def _update_context_weights(self) -> None:
        """Update context weights based on recency and importance."""
        current_time = datetime.now()
        
        for context in self.contexts:
            # Calculate temporal weight
            context_time = datetime.fromisoformat(context["timestamp"])
            age_hours = (current_time - context_time).total_seconds() / 3600
            temporal_weight = np.exp(-age_hours / 24)  # Decay over 24 hours
            
            # Calculate semantic weight
            semantic_weight = len(context["metadata"]["keywords"]) / 10  # Normalize
            
            # Calculate relationship weight
            relationship_weight = len(self.relationships.get(context["id"], set())) / 5  # Normalize
            
            # Combine weights
            total_weight = (
                self.temporal_weight * temporal_weight +
                self.semantic_weight * semantic_weight +
                self.relationship_weight * relationship_weight
            )
            
            self.context_weights[context["id"]] = total_weight

    async def _maintain_context_window(self) -> None:
        """Maintain context window by removing old contexts."""
        if len(self.contexts) > self.max_contexts:
            # Sort contexts by weight
            sorted_contexts = sorted(
                self.contexts,
                key=lambda x: self.context_weights[x["id"]]
            )
            
            # Remove contexts with lowest weights
            contexts_to_remove = sorted_contexts[:len(self.contexts) - self.max_contexts]
            for context in contexts_to_remove:
                await self._remove_context(context["id"])

    async def _remove_context(self, context_id: str) -> None:
        """Remove a context and its relationships."""
        # Remove from contexts
        self.contexts = [ctx for ctx in self.contexts if ctx["id"] != context_id]
        
        # Remove from embeddings
        context_idx = next(
            i for i, ctx in enumerate(self.contexts)
            if ctx["id"] == context_id
        )
        self.context_embeddings.pop(context_idx)
        
        # Remove relationships
        if context_id in self.relationships:
            del self.relationships[context_id]
        
        # Remove from other contexts' relationships
        for other_id in self.relationships:
            self.relationships[other_id] = {
                rel for rel in self.relationships[other_id]
                if not rel.startswith(f"{context_id}:")
            }
        
        # Remove metadata and weights
        del self.context_metadata[context_id]
        del self.context_weights[context_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all contexts."""
        messages = []
        for context in self.contexts:
            messages.extend(context["messages"])
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all contexts."""
        self.contexts = []
        self.context_embeddings = []
        self.relationships = {}
        self.context_weights = {}
        self.context_metadata = {}
        self.context_summaries = {}
        self.context_evolution = {}
        await self.save()

    async def save(self) -> None:
        """Save contexts to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "contexts": self.contexts,
                    "relationships": {
                        k: list(v) for k, v in self.relationships.items()
                    },
                    "context_weights": self.context_weights,
                    "context_metadata": {
                        k: {
                            **v,
                            "entities": list(v["entities"]),
                            "keywords": list(v["keywords"])
                        }
                        for k, v in self.context_metadata.items()
                    },
                    "context_summaries": self.context_summaries,
                    "context_evolution": self.context_evolution,
                    "last_summarization": self.last_summarization.isoformat()
                }, f)

    def load(self) -> None:
        """Load contexts from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.contexts = data.get("contexts", [])
                self.relationships = {
                    k: set(v) for k, v in data.get("relationships", {}).items()
                }
                self.context_weights = data.get("context_weights", {})
                self.context_metadata = {
                    k: {
                        **v,
                        "entities": set(v["entities"]),
                        "keywords": set(v["keywords"])
                    }
                    for k, v in data.get("context_metadata", {}).items()
                }
                self.context_summaries = data.get("context_summaries", {})
                self.context_evolution = data.get("context_evolution", {})
                self.last_summarization = datetime.fromisoformat(
                    data.get("last_summarization", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.context_embeddings = []
                for context in self.contexts:
                    context_text = ' '.join(msg['content'] for msg in context["messages"])
                    self.context_embeddings.append(self.llm.embeddings(context_text))

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_context_chain(
        self,
        context_id: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Get chain of related contexts."""
        if context_id not in self.relationships:
            return []
        
        chain = []
        visited = set()
        
        async def traverse(current_id: str, depth: int) -> None:
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            current_context = next(
                ctx for ctx in self.contexts if ctx["id"] == current_id
            )
            
            chain.append({
                "context_id": current_id,
                "messages": current_context["messages"],
                "metadata": self.context_metadata[current_id],
                "weight": self.context_weights[current_id],
                "depth": depth
            })
            
            for relationship in self.relationships[current_id]:
                related_id = relationship.split(':')[0]
                await traverse(related_id, depth + 1)
        
        await traverse(context_id, 0)
        return chain

    async def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about contexts."""
        stats = {
            "total_contexts": len(self.contexts),
            "total_messages": len(self.get_messages()),
            "relationship_types": {
                rel_type: 0 for rel_type in self.relationship_types
            },
            "topic_distribution": {},
            "sentiment_distribution": {},
            "entity_frequency": {},
            "keyword_frequency": {},
            "weight_distribution": {
                "high": 0,  # > 0.7
                "medium": 0,  # 0.3-0.7
                "low": 0  # < 0.3
            },
            "evolution_stages": {},
            "summarization_stats": {
                "summarized": len(self.context_summaries),
                "unsummarized": len(self.contexts) - len(self.context_summaries)
            }
        }
        
        for context in self.contexts:
            # Count relationship types
            if context['id'] in self.relationships:
                for relationship in self.relationships[context['id']]:
                    rel_type = relationship.split(':')[1]
                    stats["relationship_types"][rel_type] += 1
            
            # Count topics
            topic = self.context_metadata[context['id']]["topic"]
            if topic:
                stats["topic_distribution"][topic] = \
                    stats["topic_distribution"].get(topic, 0) + 1
            
            # Count sentiments
            sentiment = self.context_metadata[context['id']]["sentiment"]
            if sentiment:
                stats["sentiment_distribution"][sentiment] = \
                    stats["sentiment_distribution"].get(sentiment, 0) + 1
            
            # Count entities and keywords
            for entity in self.context_metadata[context['id']]["entities"]:
                stats["entity_frequency"][entity] = \
                    stats["entity_frequency"].get(entity, 0) + 1
            
            for keyword in self.context_metadata[context['id']]["keywords"]:
                stats["keyword_frequency"][keyword] = \
                    stats["keyword_frequency"].get(keyword, 0) + 1
            
            # Count weights
            weight = self.context_weights[context['id']]
            if weight > 0.7:
                stats["weight_distribution"]["high"] += 1
            elif weight > 0.3:
                stats["weight_distribution"]["medium"] += 1
            else:
                stats["weight_distribution"]["low"] += 1
            
            # Count evolution stages
            stage = context['metadata']['evolution_stage']
            if stage:
                stats["evolution_stages"][stage] = \
                    stats["evolution_stages"].get(stage, 0) + 1
        
        return stats

    async def get_context_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for context optimization."""
        suggestions = []
        
        # Check context count
        if len(self.contexts) > self.max_contexts * 0.8:
            suggestions.append({
                "type": "context_limit",
                "suggestion": "Consider increasing max_contexts or merging similar contexts"
            })
        
        # Check relationship distribution
        stats = await self.get_context_stats()
        for rel_type, count in stats["relationship_types"].items():
            if count == 0:
                suggestions.append({
                    "type": "relationship_diversity",
                    "suggestion": f"Consider adding more {rel_type} relationships"
                })
        
        # Check weight distribution
        if stats["weight_distribution"]["low"] > len(self.contexts) * 0.5:
            suggestions.append({
                "type": "weight_balance",
                "suggestion": "Consider adjusting weight calculation parameters"
            })
        
        # Check topic diversity
        if len(stats["topic_distribution"]) < 3:
            suggestions.append({
                "type": "topic_diversity",
                "suggestion": "Consider adding more diverse topics"
            })
        
        # Check summarization status
        if stats["summarization_stats"]["unsummarized"] > len(self.contexts) * 0.3:
            suggestions.append({
                "type": "summarization",
                "suggestion": "Consider running context summarization"
            })
        
        # Check evolution tracking
        if len(stats["evolution_stages"]) < 2:
            suggestions.append({
                "type": "evolution_tracking",
                "suggestion": "Consider enabling evolution tracking for more contexts"
            })
        
        return suggestions 