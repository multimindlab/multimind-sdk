"""
Semantic memory implementation that stores and retrieves semantic knowledge with concept relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class SemanticMemory(BaseMemory):
    """Memory that stores and retrieves semantic knowledge with concept relationships."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_concepts: int = 1000,
        similarity_threshold: float = 0.7,
        reasoning_depth: int = 3,
        concept_confidence_threshold: float = 0.6,
        enable_inference: bool = True,
        enable_validation: bool = True,
        validation_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_concepts = max_concepts
        self.similarity_threshold = similarity_threshold
        self.reasoning_depth = reasoning_depth
        self.concept_confidence_threshold = concept_confidence_threshold
        self.enable_inference = enable_inference
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        
        # Initialize concept storage
        self.concepts: List[Dict[str, Any]] = []
        self.concept_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Set[str]] = {}  # concept_id -> set of related concept_ids
        self.concept_weights: Dict[str, float] = {}  # concept_id -> weight
        self.concept_metadata: Dict[str, Dict[str, Any]] = {}  # concept_id -> metadata
        self.inference_cache: Dict[str, List[Dict[str, Any]]] = {}  # concept_id -> inferred relationships
        self.last_validation = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message as new semantic knowledge."""
        # Extract concepts from message
        concepts = await self._extract_concepts(message["content"])
        
        for concept in concepts:
            # Create concept
            concept_id = f"concept_{len(self.concepts)}"
            new_concept = {
                "id": concept_id,
                "content": concept["content"],
                "timestamp": datetime.now().isoformat(),
                "relationships": set(),
                "metadata": {
                    "category": concept["category"],
                    "properties": concept["properties"],
                    "confidence": concept["confidence"],
                    "validated": False
                }
            }
            
            # Add to storage
            self.concepts.append(new_concept)
            self.concept_weights[concept_id] = 1.0
            self.concept_metadata[concept_id] = new_concept["metadata"]
            
            # Get concept embedding
            embedding = await self.llm.embeddings(concept["content"])
            self.concept_embeddings.append(embedding)
            
            # Find related concepts
            related_concepts = await self._find_related_concepts(new_concept)
            for related in related_concepts:
                await self._add_relationship(concept_id, related["id"], related["relationship_type"])
            
            # Perform inference if enabled
            if self.enable_inference:
                await self._perform_inference(concept_id)
        
        # Check for validation
        if self.enable_validation:
            current_time = datetime.now()
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_concepts()
        
        # Maintain concept limit
        await self._maintain_concept_limit()
        
        await self.save()

    async def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts and their relationships from content."""
        try:
            prompt = f"""
            Extract semantic concepts and their relationships from the following content:
            
            Content: {content}
            
            For each concept, determine:
            1. Concept content
            2. Category
            3. Properties
            4. Confidence in extraction (0-1)
            
            Return in format:
            Concept: <concept content>
            Category: <category>
            Properties: <comma-separated properties>
            Confidence: <confidence score>
            ---
            """
            response = await self.llm.generate(prompt)
            
            concepts = []
            current_concept = {}
            
            for line in response.split('\n'):
                if line.startswith('Concept:'):
                    if current_concept:
                        concepts.append(current_concept)
                    current_concept = {
                        "content": line.split(':', 1)[1].strip(),
                        "category": None,
                        "properties": set(),
                        "confidence": 1.0
                    }
                elif line.startswith('Category:'):
                    current_concept["category"] = line.split(':', 1)[1].strip()
                elif line.startswith('Properties:'):
                    properties = line.split(':', 1)[1].strip().split(',')
                    current_concept["properties"] = {p.strip() for p in properties}
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    current_concept["confidence"] = confidence
            
            if current_concept:
                concepts.append(current_concept)
            
            return concepts
            
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            return []

    async def _find_related_concepts(
        self,
        concept: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find concepts related to the given concept."""
        if not self.concepts:
            return []
        
        # Get concept embedding
        concept_embedding = await self.llm.embeddings(concept["content"])
        
        # Calculate similarities
        similarities = []
        for i, existing_embedding in enumerate(self.concept_embeddings):
            similarity = self._cosine_similarity(concept_embedding, existing_embedding)
            if similarity >= self.similarity_threshold:
                similarities.append({
                    "id": self.concepts[i]["id"],
                    "similarity": similarity,
                    "relationship_type": await self._determine_relationship_type(
                        concept,
                        self.concepts[i]
                    )
                })
        
        return sorted(similarities, key=lambda x: x["similarity"], reverse=True)

    async def _determine_relationship_type(
        self,
        concept1: Dict[str, Any],
        concept2: Dict[str, Any]
    ) -> str:
        """Determine the type of relationship between two concepts."""
        try:
            prompt = f"""
            Determine the relationship type between these concepts:
            
            Concept 1: {concept1['content']}
            Category: {concept1['metadata']['category']}
            Properties: {concept1['metadata']['properties']}
            
            Concept 2: {concept2['content']}
            Category: {concept2['metadata']['category']}
            Properties: {concept2['metadata']['properties']}
            
            Choose from: is_a, part_of, has_property, related_to, contradicts, supports
            """
            response = await self.llm.generate(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Error determining relationship type: {e}")
            return "related_to"

    async def _add_relationship(
        self,
        concept_id1: str,
        concept_id2: str,
        relationship_type: str
    ) -> None:
        """Add relationship between two concepts."""
        if concept_id1 not in self.relationships:
            self.relationships[concept_id1] = set()
        if concept_id2 not in self.relationships:
            self.relationships[concept_id2] = set()
        
        self.relationships[concept_id1].add(f"{concept_id2}:{relationship_type}")
        self.relationships[concept_id2].add(f"{concept_id1}:{relationship_type}")

    async def _perform_inference(self, concept_id: str) -> None:
        """Perform inference to discover new relationships."""
        if concept_id not in self.inference_cache:
            self.inference_cache[concept_id] = []
        
        try:
            # Get concept and its relationships
            concept = next(c for c in self.concepts if c["id"] == concept_id)
            relationships = self.relationships.get(concept_id, set())
            
            # Generate inference prompt
            prompt = f"""
            Based on this concept and its relationships, infer new relationships:
            
            Concept: {concept['content']}
            Category: {concept['metadata']['category']}
            Properties: {concept['metadata']['properties']}
            Current Relationships: {relationships}
            
            Return inferred relationships in format:
            Related Concept: <concept content>
            Relationship Type: <relationship type>
            Confidence: <confidence score>
            ---
            """
            response = await self.llm.generate(prompt)
            
            # Parse inferred relationships
            current_relationship = {}
            for line in response.split('\n'):
                if line.startswith('Related Concept:'):
                    if current_relationship:
                        self.inference_cache[concept_id].append(current_relationship)
                    current_relationship = {
                        "concept": line.split(':', 1)[1].strip(),
                        "relationship_type": None,
                        "confidence": None
                    }
                elif line.startswith('Relationship Type:'):
                    current_relationship["relationship_type"] = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    current_relationship["confidence"] = confidence
            
            if current_relationship:
                self.inference_cache[concept_id].append(current_relationship)
            
        except Exception as e:
            print(f"Error performing inference: {e}")

    async def _validate_concepts(self) -> None:
        """Validate concepts and their relationships."""
        for concept in self.concepts:
            if concept["metadata"]["validated"]:
                continue
            
            try:
                # Generate validation prompt
                prompt = f"""
                Validate this concept and its relationships:
                
                Concept: {concept['content']}
                Category: {concept['metadata']['category']}
                Properties: {concept['metadata']['properties']}
                Relationships: {self.relationships.get(concept['id'], set())}
                
                Return validation results in format:
                Valid: <true/false>
                Confidence: <confidence score>
                Issues: <comma-separated issues>
                """
                response = await self.llm.generate(prompt)
                
                # Parse validation results
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('Valid:'):
                        is_valid = line.split(':', 1)[1].strip().lower() == 'true'
                    elif line.startswith('Confidence:'):
                        confidence = float(line.split(':', 1)[1].strip())
                    elif line.startswith('Issues:'):
                        issues = line.split(':', 1)[1].strip().split(',')
                
                if is_valid and confidence >= self.concept_confidence_threshold:
                    concept["metadata"]["validated"] = True
                    concept["metadata"]["confidence"] = confidence
                else:
                    # Remove invalid concept
                    await self._remove_concept(concept["id"])
            
            except Exception as e:
                print(f"Error validating concept: {e}")
        
        self.last_validation = datetime.now()

    async def _maintain_concept_limit(self) -> None:
        """Maintain concept limit by removing least important concepts."""
        if len(self.concepts) > self.max_concepts:
            # Sort concepts by weight
            sorted_concepts = sorted(
                self.concepts,
                key=lambda x: self.concept_weights[x["id"]]
            )
            
            # Remove concepts with lowest weights
            concepts_to_remove = sorted_concepts[:len(self.concepts) - self.max_concepts]
            for concept in concepts_to_remove:
                await self._remove_concept(concept["id"])

    async def _remove_concept(self, concept_id: str) -> None:
        """Remove a concept and its relationships."""
        # Remove from concepts
        concept_idx = next(i for i, c in enumerate(self.concepts) if c["id"] == concept_id)
        self.concepts.pop(concept_idx)
        self.concept_embeddings.pop(concept_idx)
        
        # Remove relationships
        if concept_id in self.relationships:
            del self.relationships[concept_id]
        
        # Remove from other concepts' relationships
        for other_id in self.relationships:
            self.relationships[other_id] = {
                rel for rel in self.relationships[other_id]
                if not rel.startswith(f"{concept_id}:")
            }
        
        # Remove metadata and weights
        del self.concept_metadata[concept_id]
        del self.concept_weights[concept_id]
        
        # Remove from inference cache
        if concept_id in self.inference_cache:
            del self.inference_cache[concept_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all concepts."""
        messages = []
        for concept in self.concepts:
            messages.append({
                "role": "concept",
                "content": concept["content"],
                "timestamp": concept["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all concepts."""
        self.concepts = []
        self.concept_embeddings = []
        self.relationships = {}
        self.concept_weights = {}
        self.concept_metadata = {}
        self.inference_cache = {}
        await self.save()

    async def save(self) -> None:
        """Save concepts to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "concepts": self.concepts,
                    "relationships": {
                        k: list(v) for k, v in self.relationships.items()
                    },
                    "concept_weights": self.concept_weights,
                    "concept_metadata": {
                        k: {
                            **v,
                            "properties": list(v["properties"])
                        }
                        for k, v in self.concept_metadata.items()
                    },
                    "inference_cache": self.inference_cache,
                    "last_validation": self.last_validation.isoformat()
                }, f)

    def load(self) -> None:
        """Load concepts from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.concepts = data.get("concepts", [])
                self.relationships = {
                    k: set(v) for k, v in data.get("relationships", {}).items()
                }
                self.concept_weights = data.get("concept_weights", {})
                self.concept_metadata = {
                    k: {
                        **v,
                        "properties": set(v["properties"])
                    }
                    for k, v in data.get("concept_metadata", {}).items()
                }
                self.inference_cache = data.get("inference_cache", {})
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.concept_embeddings = []
                for concept in self.concepts:
                    self.concept_embeddings.append(
                        self.llm.embeddings(concept["content"])
                    )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by its ID."""
        try:
            return next(c for c in self.concepts if c["id"] == concept_id)
        except StopIteration:
            return None

    async def get_related_concepts(
        self,
        concept_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get concepts related to the given concept."""
        if concept_id not in self.relationships:
            return []
        
        related_concepts = []
        for relationship in self.relationships[concept_id]:
            related_id, rel_type = relationship.split(':')
            if relationship_type is None or rel_type == relationship_type:
                concept = await self.get_concept_by_id(related_id)
                if concept:
                    related_concepts.append({
                        "concept": concept,
                        "relationship_type": rel_type
                    })
        
        return related_concepts

    async def get_inferred_relationships(
        self,
        concept_id: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get inferred relationships for a concept."""
        if concept_id not in self.inference_cache:
            return []
        
        if min_confidence is None:
            return self.inference_cache[concept_id]
        
        return [
            rel for rel in self.inference_cache[concept_id]
            if rel["confidence"] >= min_confidence
        ]

    async def get_concept_stats(self) -> Dict[str, Any]:
        """Get statistics about concepts."""
        stats = {
            "total_concepts": len(self.concepts),
            "category_distribution": {},
            "relationship_types": {
                rel_type: 0 for rel_type in [
                    "is_a", "part_of", "has_property",
                    "related_to", "contradicts", "supports"
                ]
            },
            "confidence_distribution": {
                "high": 0,  # > 0.8
                "medium": 0,  # 0.5-0.8
                "low": 0  # < 0.5
            },
            "validation_stats": {
                "validated": 0,
                "unvalidated": 0
            },
            "inference_stats": {
                "concepts_with_inferences": len(self.inference_cache),
                "total_inferences": sum(len(inferences) for inferences in self.inference_cache.values())
            }
        }
        
        for concept in self.concepts:
            # Count categories
            category = concept["metadata"]["category"]
            if category:
                stats["category_distribution"][category] = \
                    stats["category_distribution"].get(category, 0) + 1
            
            # Count relationship types
            if concept["id"] in self.relationships:
                for relationship in self.relationships[concept["id"]]:
                    rel_type = relationship.split(':')[1]
                    stats["relationship_types"][rel_type] += 1
            
            # Count confidence levels
            confidence = concept["metadata"]["confidence"]
            if confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif confidence > 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
            
            # Count validation status
            if concept["metadata"]["validated"]:
                stats["validation_stats"]["validated"] += 1
            else:
                stats["validation_stats"]["unvalidated"] += 1
        
        return stats

    async def get_concept_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for concept optimization."""
        suggestions = []
        
        # Check concept count
        if len(self.concepts) > self.max_concepts * 0.8:
            suggestions.append({
                "type": "concept_limit",
                "suggestion": "Consider increasing max_concepts or removing less important concepts"
            })
        
        # Check confidence distribution
        stats = await self.get_concept_stats()
        if stats["confidence_distribution"]["low"] > len(self.concepts) * 0.3:
            suggestions.append({
                "type": "confidence_quality",
                "suggestion": "Consider improving concept extraction quality"
            })
        
        # Check validation status
        if stats["validation_stats"]["unvalidated"] > len(self.concepts) * 0.5:
            suggestions.append({
                "type": "validation",
                "suggestion": "Consider running concept validation"
            })
        
        # Check relationship diversity
        if len(stats["relationship_types"]) < 3:
            suggestions.append({
                "type": "relationship_diversity",
                "suggestion": "Consider adding more diverse relationship types"
            })
        
        # Check inference coverage
        if stats["inference_stats"]["concepts_with_inferences"] < len(self.concepts) * 0.5:
            suggestions.append({
                "type": "inference_coverage",
                "suggestion": "Consider performing inference on more concepts"
            })
        
        return suggestions 