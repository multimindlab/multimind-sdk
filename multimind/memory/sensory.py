"""
Sensory memory implementation that manages sensory experiences across different modalities.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class SensoryMemory(BaseMemory):
    """Memory that manages sensory experiences across different modalities."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_experiences: int = 1000,
        sensory_threshold: float = 0.7,
        enable_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_relationships: bool = True,
        relationship_interval: int = 3600,  # 1 hour
        enable_patterns: bool = True,
        pattern_interval: int = 3600,  # 1 hour
        enable_learning: bool = True,
        learning_rate: float = 0.1,
        enable_history: bool = True,
        history_window: int = 100,
        enable_evolution: bool = True,
        evolution_interval: int = 3600,  # 1 hour
        enable_validation: bool = True,
        validation_interval: int = 3600,  # 1 hour
        enable_cross_modal: bool = True,
        cross_modal_interval: int = 3600,  # 1 hour
        enable_fusion: bool = True,
        fusion_interval: int = 3600,  # 1 hour
        enable_advanced_patterns: bool = True,
        advanced_pattern_interval: int = 3600,  # 1 hour
        relationship_types: Set[str] = None,
        modalities: Set[str] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_experiences = max_experiences
        self.sensory_threshold = sensory_threshold
        self.enable_analysis = enable_analysis
        self.analysis_interval = analysis_interval
        self.enable_relationships = enable_relationships
        self.relationship_interval = relationship_interval
        self.enable_patterns = enable_patterns
        self.pattern_interval = pattern_interval
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.enable_history = enable_history
        self.history_window = history_window
        self.enable_evolution = enable_evolution
        self.evolution_interval = evolution_interval
        self.enable_validation = enable_validation
        self.validation_interval = validation_interval
        self.enable_cross_modal = enable_cross_modal
        self.cross_modal_interval = cross_modal_interval
        self.enable_fusion = enable_fusion
        self.fusion_interval = fusion_interval
        self.enable_advanced_patterns = enable_advanced_patterns
        self.advanced_pattern_interval = advanced_pattern_interval
        self.relationship_types = relationship_types or {
            "complements",
            "enhances",
            "conflicts",
            "triggers",
            "inhibits",
            "modulates",
            "synchronizes",
            "precedes",
            "follows",
            "co_occurs"
        }
        self.modalities = modalities or {
            "visual",
            "auditory",
            "tactile",
            "olfactory",
            "gustatory",
            "proprioceptive",
            "vestibular",
            "interoceptive"
        }
        
        # Initialize sensory memory storage
        self.experiences: List[Dict[str, Any]] = []
        self.experience_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # experience_id -> {relationship_type -> target_ids}
        self.patterns: Dict[str, List[str]] = {}  # pattern_id -> experience_ids
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # experience_id -> learning records
        self.experience_history: List[Dict[str, Any]] = []  # Recent experience updates
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # experience_id -> evolution records
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}  # experience_id -> validation records
        self.cross_modal_links: Dict[str, Dict[str, List[str]]] = {}  # experience_id -> {modality -> related_ids}
        self.fused_experiences: Dict[str, Dict[str, Any]] = {}  # fused_id -> fused experience data
        self.advanced_patterns: Dict[str, Dict[str, Any]] = {}  # pattern_id -> pattern data
        self.last_analysis = datetime.now()
        self.last_relationship_update = datetime.now()
        self.last_pattern_update = datetime.now()
        self.last_evolution = datetime.now()
        self.last_validation = datetime.now()
        self.last_cross_modal = datetime.now()
        self.last_fusion = datetime.now()
        self.last_advanced_pattern = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and analyze sensory information."""
        # Create new experience
        experience_id = f"experience_{len(self.experiences)}"
        new_experience = {
            "id": experience_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "modalities": [],
                "intensity": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "duration": None,
                "location": None,
                "context": None,
                "learning_progress": 0.0,
                "evolution_stage": 0,
                "validation_score": 0.0,
                "analysis_results": {},
                "validation_results": {},
                "cross_modal_links": {},
                "fusion_data": {},
                "pattern_membership": []
            }
        }
        
        # Add to storage
        self.experiences.append(new_experience)
        
        # Get experience embedding
        embedding = await self.llm.embeddings(message["content"])
        self.experience_embeddings.append(embedding)
        
        # Analyze sensory information
        if self.enable_analysis:
            current_time = datetime.now()
            if (current_time - self.last_analysis).total_seconds() > self.analysis_interval:
                await self._analyze_sensory_info(experience_id)
        
        # Find relationships
        if self.enable_relationships:
            current_time = datetime.now()
            if (current_time - self.last_relationship_update).total_seconds() > self.relationship_interval:
                await self._find_relationships(experience_id)
        
        # Update patterns
        if self.enable_patterns:
            current_time = datetime.now()
            if (current_time - self.last_pattern_update).total_seconds() > self.pattern_interval:
                await self._update_patterns()
        
        # Update cross-modal links
        if self.enable_cross_modal:
            current_time = datetime.now()
            if (current_time - self.last_cross_modal).total_seconds() > self.cross_modal_interval:
                await self._update_cross_modal_links(experience_id)
        
        # Update sensory fusion
        if self.enable_fusion:
            current_time = datetime.now()
            if (current_time - self.last_fusion).total_seconds() > self.fusion_interval:
                await self._update_sensory_fusion(experience_id)
        
        # Update advanced patterns
        if self.enable_advanced_patterns:
            current_time = datetime.now()
            if (current_time - self.last_advanced_pattern).total_seconds() > self.advanced_pattern_interval:
                await self._update_advanced_patterns()
        
        # Update experience history
        if self.enable_history:
            self.experience_history.append({
                "experience_id": experience_id,
                "timestamp": new_experience["timestamp"],
                "content": new_experience["content"],
                "modalities": new_experience["metadata"]["modalities"],
                "intensity": new_experience["metadata"]["intensity"],
                "valence": new_experience["metadata"]["valence"],
                "arousal": new_experience["metadata"]["arousal"]
            })
            if len(self.experience_history) > self.history_window:
                self.experience_history.pop(0)
        
        # Update learning progress
        if self.enable_learning:
            await self._update_learning_progress(experience_id)
        
        # Update evolution
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(experience_id)
        
        # Validate experience
        if self.enable_validation:
            current_time = datetime.now()
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_experience(experience_id)
        
        # Maintain experience limit
        await self._maintain_experience_limit()
        
        await self.save()

    async def _analyze_sensory_info(self, experience_id: str) -> None:
        """Analyze sensory information from a message."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        try:
            # Generate analysis prompt
            prompt = f"""
            Analyze the sensory information in this message:
            
            {experience['content']}
            
            Return a JSON object with:
            1. modalities: list of strings (e.g., visual, auditory, tactile)
            2. intensity: float (0-1)
            3. valence: float (-1 to 1)
            4. arousal: float (0-1)
            5. duration: string (e.g., "2 seconds") or null
            6. location: string or null
            7. context: string or null
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)
            
            # Update experience metadata
            experience["metadata"]["modalities"] = analysis.get("modalities", [])
            experience["metadata"]["intensity"] = analysis.get("intensity", 0.0)
            experience["metadata"]["valence"] = analysis.get("valence", 0.0)
            experience["metadata"]["arousal"] = analysis.get("arousal", 0.0)
            experience["metadata"]["duration"] = analysis.get("duration")
            experience["metadata"]["location"] = analysis.get("location")
            experience["metadata"]["context"] = analysis.get("context")
            experience["metadata"]["analysis_results"] = analysis
            
        except Exception as e:
            print(f"Error analyzing sensory info: {e}")

    async def _find_relationships(self, experience_id: str) -> None:
        """Find relationships between sensory experiences."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        for other_experience in self.experiences:
            if other_experience["id"] == experience_id:
                continue
            
            # Calculate sensory similarity
            similarity = self._calculate_sensory_similarity(
                experience["metadata"],
                other_experience["metadata"]
            )
            
            if similarity >= self.sensory_threshold:
                # Determine relationship type
                relationship_type = await self._determine_relationship_type(
                    experience,
                    other_experience,
                    similarity
                )
                
                if relationship_type:
                    # Add bidirectional relationship
                    self.relationships[experience_id][relationship_type].append(other_experience["id"])
                    self.relationships[other_experience["id"]][relationship_type].append(experience_id)

    def _calculate_sensory_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two sensory experiences."""
        # Calculate modality similarity
        modality_similarity = len(
            set(metadata1["modalities"]) & set(metadata2["modalities"])
        ) / len(
            set(metadata1["modalities"]) | set(metadata2["modalities"])
        ) if metadata1["modalities"] and metadata2["modalities"] else 0.0
        
        # Calculate intensity similarity
        intensity_similarity = 1.0 - abs(metadata1["intensity"] - metadata2["intensity"])
        
        # Calculate valence similarity
        valence_similarity = 1.0 - abs(metadata1["valence"] - metadata2["valence"]) / 2.0
        
        # Calculate arousal similarity
        arousal_similarity = 1.0 - abs(metadata1["arousal"] - metadata2["arousal"])
        
        # Calculate location similarity if available
        location_similarity = 1.0 if metadata1["location"] == metadata2["location"] else 0.0
        
        # Calculate context similarity if available
        context_similarity = 1.0 if metadata1["context"] == metadata2["context"] else 0.0
        
        return (
            modality_similarity * 0.3 +
            intensity_similarity * 0.2 +
            valence_similarity * 0.2 +
            arousal_similarity * 0.2 +
            location_similarity * 0.05 +
            context_similarity * 0.05
        )

    async def _determine_relationship_type(
        self,
        experience1: Dict[str, Any],
        experience2: Dict[str, Any],
        similarity: float
    ) -> Optional[str]:
        """Determine the type of relationship between two sensory experiences."""
        try:
            prompt = f"""
            Determine the relationship type between these two sensory experiences:
            
            Experience 1: {experience1['content']}
            Modalities: {', '.join(experience1['metadata']['modalities'])}
            Intensity: {experience1['metadata']['intensity']}
            Valence: {experience1['metadata']['valence']}
            Arousal: {experience1['metadata']['arousal']}
            Location: {experience1['metadata']['location']}
            Context: {experience1['metadata']['context']}
            
            Experience 2: {experience2['content']}
            Modalities: {', '.join(experience2['metadata']['modalities'])}
            Intensity: {experience2['metadata']['intensity']}
            Valence: {experience2['metadata']['valence']}
            Arousal: {experience2['metadata']['arousal']}
            Location: {experience2['metadata']['location']}
            Context: {experience2['metadata']['context']}
            
            Similarity: {similarity}
            
            Available relationship types: {', '.join(self.relationship_types)}
            
            Return the most appropriate relationship type or 'none' if no clear relationship exists.
            """
            response = await self.llm.generate(prompt)
            
            relationship_type = response.strip().lower()
            if relationship_type in self.relationship_types:
                return relationship_type
            
            return None
            
        except Exception as e:
            print(f"Error determining relationship type: {e}")
            return None

    async def _update_patterns(self) -> None:
        """Update patterns of related experiences."""
        # Clear existing patterns
        self.patterns = {}
        
        # Group by relationship types
        for relationship_type in self.relationship_types:
            # Find connected components
            visited = set()
            
            for experience_id in self.relationships:
                if experience_id in visited:
                    continue
                
                # Start new pattern
                pattern_id = f"pattern_{len(self.patterns)}"
                pattern = []
                
                # DFS to find connected experiences
                stack = [experience_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    pattern.append(current_id)
                    
                    # Add related experiences
                    for related_id in self.relationships[current_id][relationship_type]:
                        if related_id not in visited:
                            stack.append(related_id)
                
                if len(pattern) >= 2:  # Minimum pattern size
                    self.patterns[pattern_id] = pattern
        
        self.last_pattern_update = datetime.now()

    async def _update_learning_progress(self, experience_id: str) -> None:
        """Update learning progress for an experience."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        # Calculate learning metrics
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[experience_id].values()
        )
        intensity = experience["metadata"]["intensity"]
        validation_score = experience["metadata"]["validation_score"]
        
        # Update learning progress
        progress = (
            self.learning_rate * (relationship_count / len(self.relationship_types)) +
            self.learning_rate * intensity +
            self.learning_rate * validation_score
        )
        
        experience["metadata"]["learning_progress"] = min(
            1.0,
            experience["metadata"]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[experience_id].append({
            "timestamp": datetime.now().isoformat(),
            "relationship_count": relationship_count,
            "intensity": intensity,
            "validation_score": validation_score,
            "progress": progress
        })

    async def _update_evolution(self, experience_id: str) -> None:
        """Update evolution stage for an experience."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        # Calculate evolution metrics
        learning_progress = experience["metadata"]["learning_progress"]
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[experience_id].values()
        )
        validation_score = experience["metadata"]["validation_score"]
        
        # Determine evolution stage
        if learning_progress >= 0.8 and validation_score >= 0.8:
            stage = 3  # Mature
        elif learning_progress >= 0.5 or validation_score >= 0.5:
            stage = 2  # Developing
        elif relationship_count > 0:
            stage = 1  # Emerging
        else:
            stage = 0  # New
        
        # Update evolution stage
        experience["metadata"]["evolution_stage"] = stage
        
        # Record evolution
        self.evolution_history[experience_id].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "learning_progress": learning_progress,
            "relationship_count": relationship_count,
            "validation_score": validation_score
        })

    async def _validate_experience(self, experience_id: str) -> None:
        """Validate sensory information of an experience."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        try:
            # Generate validation prompt
            prompt = f"""
            Validate the sensory information of this experience:
            
            {experience['content']}
            
            Modalities: {', '.join(experience['metadata']['modalities'])}
            Intensity: {experience['metadata']['intensity']}
            Valence: {experience['metadata']['valence']}
            Arousal: {experience['metadata']['arousal']}
            Duration: {experience['metadata']['duration']}
            Location: {experience['metadata']['location']}
            Context: {experience['metadata']['context']}
            
            Return a JSON object with:
            1. validation_score: float (0-1)
            2. validation_reason: string
            3. inconsistencies: list of strings
            4. suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            validation = json.loads(response)
            
            # Update experience metadata
            experience["metadata"]["validation_score"] = validation["validation_score"]
            experience["metadata"]["validation_results"] = validation
            
            # Record validation
            self.validation_history[experience_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": validation["validation_score"],
                "reason": validation["validation_reason"],
                "inconsistencies": validation["inconsistencies"],
                "suggestions": validation["suggestions"]
            })
            
        except Exception as e:
            print(f"Error validating experience: {e}")

    async def _maintain_experience_limit(self) -> None:
        """Maintain experience limit by removing least important experiences."""
        if len(self.experiences) > self.max_experiences:
            # Sort experiences by learning progress and validation score
            sorted_experiences = sorted(
                self.experiences,
                key=lambda x: (
                    x["metadata"]["learning_progress"] +
                    x["metadata"]["validation_score"]
                )
            )
            
            # Remove experiences with lowest scores
            experiences_to_remove = sorted_experiences[:len(self.experiences) - self.max_experiences]
            for experience in experiences_to_remove:
                await self._remove_experience(experience["id"])

    async def _remove_experience(self, experience_id: str) -> None:
        """Remove an experience and its associated data."""
        # Remove from experiences
        experience_idx = next(i for i, e in enumerate(self.experiences) if e["id"] == experience_id)
        self.experiences.pop(experience_idx)
        self.experience_embeddings.pop(experience_idx)
        
        # Remove from relationships
        if experience_id in self.relationships:
            del self.relationships[experience_id]
        
        # Remove from patterns
        for pattern_id, pattern in self.patterns.items():
            if experience_id in pattern:
                pattern.remove(experience_id)
                if len(pattern) < 2:  # Minimum pattern size
                    del self.patterns[pattern_id]
        
        # Remove from history
        if self.enable_history:
            self.experience_history = [
                e for e in self.experience_history
                if e["experience_id"] != experience_id
            ]
        
        # Remove learning history
        if experience_id in self.learning_history:
            del self.learning_history[experience_id]
        
        # Remove evolution history
        if experience_id in self.evolution_history:
            del self.evolution_history[experience_id]
        
        # Remove validation history
        if experience_id in self.validation_history:
            del self.validation_history[experience_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all experiences."""
        messages = []
        for experience in self.experiences:
            messages.append({
                "role": "sensory_memory",
                "content": experience["content"],
                "timestamp": experience["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all experiences."""
        self.experiences = []
        self.experience_embeddings = []
        self.relationships = {}
        self.patterns = {}
        self.learning_history = {}
        self.experience_history = []
        self.evolution_history = {}
        self.validation_history = {}
        await self.save()

    async def save(self) -> None:
        """Save experiences to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "experiences": self.experiences,
                    "relationships": self.relationships,
                    "patterns": self.patterns,
                    "learning_history": self.learning_history,
                    "experience_history": self.experience_history,
                    "evolution_history": self.evolution_history,
                    "validation_history": self.validation_history,
                    "cross_modal_links": self.cross_modal_links,
                    "fused_experiences": self.fused_experiences,
                    "advanced_patterns": self.advanced_patterns,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_relationship_update": self.last_relationship_update.isoformat(),
                    "last_pattern_update": self.last_pattern_update.isoformat(),
                    "last_evolution": self.last_evolution.isoformat(),
                    "last_validation": self.last_validation.isoformat(),
                    "last_cross_modal": self.last_cross_modal.isoformat(),
                    "last_fusion": self.last_fusion.isoformat(),
                    "last_advanced_pattern": self.last_advanced_pattern.isoformat()
                }, f)

    def load(self) -> None:
        """Load experiences from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.experiences = data.get("experiences", [])
                self.relationships = data.get("relationships", {})
                self.patterns = data.get("patterns", {})
                self.learning_history = data.get("learning_history", {})
                self.experience_history = data.get("experience_history", [])
                self.evolution_history = data.get("evolution_history", {})
                self.validation_history = data.get("validation_history", {})
                self.cross_modal_links = data.get("cross_modal_links", {})
                self.fused_experiences = data.get("fused_experiences", {})
                self.advanced_patterns = data.get("advanced_patterns", {})
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_relationship_update = datetime.fromisoformat(
                    data.get("last_relationship_update", datetime.now().isoformat())
                )
                self.last_pattern_update = datetime.fromisoformat(
                    data.get("last_pattern_update", datetime.now().isoformat())
                )
                self.last_evolution = datetime.fromisoformat(
                    data.get("last_evolution", datetime.now().isoformat())
                )
                self.last_validation = datetime.fromisoformat(
                    data.get("last_validation", datetime.now().isoformat())
                )
                self.last_cross_modal = datetime.fromisoformat(
                    data.get("last_cross_modal", datetime.now().isoformat())
                )
                self.last_fusion = datetime.fromisoformat(
                    data.get("last_fusion", datetime.now().isoformat())
                )
                self.last_advanced_pattern = datetime.fromisoformat(
                    data.get("last_advanced_pattern", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.experience_embeddings = []
                for experience in self.experiences:
                    self.experience_embeddings.append(
                        self.llm.embeddings(experience["content"])
                    )

    async def get_sensory_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about sensory memory."""
        stats = {
            "total_experiences": len(self.experiences),
            "modality_distribution": {
                modality: sum(
                    1 for e in self.experiences
                    if modality in e["metadata"]["modalities"]
                )
                for modality in self.modalities
            },
            "relationship_stats": {
                "total_relationships": sum(
                    len(relationships)
                    for relationships in self.relationships.values()
                ),
                "relationship_types": {
                    rel_type: sum(
                        1 for relationships in self.relationships.values()
                        if relationships[rel_type]
                    )
                    for rel_type in self.relationship_types
                }
            },
            "pattern_stats": {
                "total_patterns": len(self.patterns),
                "average_pattern_size": sum(len(pattern) for pattern in self.patterns.values()) / len(self.patterns) if self.patterns else 0,
                "max_pattern_size": max(len(pattern) for pattern in self.patterns.values()) if self.patterns else 0
            },
            "learning_stats": {
                "average_progress": sum(
                    e["metadata"]["learning_progress"]
                    for e in self.experiences
                ) / len(self.experiences) if self.experiences else 0,
                "experiences_with_progress": sum(
                    1 for e in self.experiences
                    if e["metadata"]["learning_progress"] > 0
                )
            },
            "evolution_stats": {
                "stage_distribution": {
                    stage: sum(1 for e in self.experiences if e["metadata"]["evolution_stage"] == stage)
                    for stage in range(4)
                },
                "average_stage": sum(e["metadata"]["evolution_stage"] for e in self.experiences) / len(self.experiences) if self.experiences else 0
            },
            "validation_stats": {
                "average_score": sum(
                    e["metadata"]["validation_score"]
                    for e in self.experiences
                ) / len(self.experiences) if self.experiences else 0,
                "validated_experiences": sum(
                    1 for e in self.experiences
                    if e["metadata"]["validation_score"] >= 0.8
                )
            }
        }
        
        # Add cross-modal statistics
        if self.enable_cross_modal:
            stats["cross_modal_stats"] = {
                "total_links": sum(
                    len(links)
                    for links in self.cross_modal_links.values()
                ),
                "modality_distribution": {
                    modality: sum(
                        1 for links in self.cross_modal_links.values()
                        if links[modality]
                    )
                    for modality in self.modalities
                }
            }
        
        # Add fusion statistics
        if self.enable_fusion:
            stats["fusion_stats"] = {
                "total_fused": len(self.fused_experiences),
                "fusion_types": {
                    fused["metadata"]["fusion_type"]: sum(
                        1 for f in self.fused_experiences.values()
                        if f["metadata"]["fusion_type"] == fused["metadata"]["fusion_type"]
                    )
                    for fused in self.fused_experiences.values()
                },
                "average_confidence": sum(
                    fused["metadata"]["confidence"]
                    for fused in self.fused_experiences.values()
                ) / len(self.fused_experiences) if self.fused_experiences else 0
            }
        
        # Add advanced pattern statistics
        if self.enable_advanced_patterns:
            stats["advanced_pattern_stats"] = {
                "total_patterns": len(self.advanced_patterns),
                "pattern_types": {
                    pattern["type"]: sum(
                        1 for p in self.advanced_patterns.values()
                        if p["type"] == pattern["type"]
                    )
                    for pattern in self.advanced_patterns.values()
                },
                "average_pattern_size": sum(
                    len(pattern["experiences"])
                    for pattern in self.advanced_patterns.values()
                ) / len(self.advanced_patterns) if self.advanced_patterns else 0
            }
        
        return stats

    async def get_sensory_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for sensory memory optimization."""
        suggestions = []
        
        # Check experience count
        if len(self.experiences) > self.max_experiences * 0.8:
            suggestions.append({
                "type": "experience_limit",
                "suggestion": "Consider increasing max_experiences or removing less important experiences"
            })
        
        # Check relationship quality
        stats = await self.get_sensory_memory_stats()
        if stats["relationship_stats"]["total_relationships"] < len(self.experiences) * 2:
            suggestions.append({
                "type": "relationship_development",
                "suggestion": "Consider developing more sensory relationships between experiences"
            })
        
        # Check pattern quality
        if stats["pattern_stats"]["average_pattern_size"] < 2:
            suggestions.append({
                "type": "pattern_development",
                "suggestion": "Consider developing more sensory patterns or adjusting pattern detection"
            })
        
        # Check learning progress
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms for experiences"
            })
        
        # Check evolution progress
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for experiences"
            })
        
        # Check validation quality
        if stats["validation_stats"]["average_score"] < 0.8:
            suggestions.append({
                "type": "validation_improvement",
                "suggestion": "Consider improving validation mechanisms or resolving inconsistencies"
            })
        
        # Add cross-modal suggestions
        if self.enable_cross_modal:
            if stats["cross_modal_stats"]["total_links"] < len(self.experiences):
                suggestions.append({
                    "type": "cross_modal_development",
                    "suggestion": "Consider developing more cross-modal links between experiences"
                })
        
        # Add fusion suggestions
        if self.enable_fusion:
            if stats["fusion_stats"]["total_fused"] < len(self.experiences) * 0.1:
                suggestions.append({
                    "type": "fusion_development",
                    "suggestion": "Consider developing more fused experiences"
                })
        
        # Add advanced pattern suggestions
        if self.enable_advanced_patterns:
            if stats["advanced_pattern_stats"]["total_patterns"] < len(self.experiences) * 0.05:
                suggestions.append({
                    "type": "pattern_development",
                    "suggestion": "Consider developing more advanced patterns"
                })
        
        return suggestions

    async def _update_cross_modal_links(self, experience_id: str) -> None:
        """Update cross-modal links between experiences."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        # Initialize cross-modal links for this experience
        self.cross_modal_links[experience_id] = {
            modality: [] for modality in self.modalities
        }
        
        for other_experience in self.experiences:
            if other_experience["id"] == experience_id:
                continue
            
            # Find complementary modalities
            experience_modalities = set(experience["metadata"]["modalities"])
            other_modalities = set(other_experience["metadata"]["modalities"])
            
            # Check for complementary modalities
            for modality in experience_modalities:
                if modality not in other_modalities:
                    # Calculate cross-modal similarity
                    similarity = self._calculate_cross_modal_similarity(
                        experience,
                        other_experience,
                        modality
                    )
                    
                    if similarity >= self.sensory_threshold:
                        self.cross_modal_links[experience_id][modality].append(other_experience["id"])
        
        # Update experience metadata
        experience["metadata"]["cross_modal_links"] = {
            modality: len(links)
            for modality, links in self.cross_modal_links[experience_id].items()
        }
        
        self.last_cross_modal = datetime.now()

    def _calculate_cross_modal_similarity(
        self,
        experience1: Dict[str, Any],
        experience2: Dict[str, Any],
        modality: str
    ) -> float:
        """Calculate similarity between experiences across different modalities."""
        # Calculate temporal similarity
        time1 = datetime.fromisoformat(experience1["timestamp"])
        time2 = datetime.fromisoformat(experience2["timestamp"])
        temporal_similarity = 1.0 / (1.0 + abs((time1 - time2).total_seconds()))
        
        # Calculate intensity similarity
        intensity_similarity = 1.0 - abs(
            experience1["metadata"]["intensity"] -
            experience2["metadata"]["intensity"]
        )
        
        # Calculate valence similarity
        valence_similarity = 1.0 - abs(
            experience1["metadata"]["valence"] -
            experience2["metadata"]["valence"]
        ) / 2.0
        
        # Calculate arousal similarity
        arousal_similarity = 1.0 - abs(
            experience1["metadata"]["arousal"] -
            experience2["metadata"]["arousal"]
        )
        
        # Calculate location similarity if available
        location_similarity = 1.0 if (
            experience1["metadata"]["location"] == experience2["metadata"]["location"]
        ) else 0.0
        
        # Calculate context similarity if available
        context_similarity = 1.0 if (
            experience1["metadata"]["context"] == experience2["metadata"]["context"]
        ) else 0.0
        
        return (
            temporal_similarity * 0.3 +
            intensity_similarity * 0.2 +
            valence_similarity * 0.2 +
            arousal_similarity * 0.2 +
            location_similarity * 0.05 +
            context_similarity * 0.05
        )

    async def _update_sensory_fusion(self, experience_id: str) -> None:
        """Update sensory fusion for experiences."""
        experience = next(e for e in self.experiences if e["id"] == experience_id)
        
        # Find experiences to fuse with
        fusion_candidates = []
        for other_experience in self.experiences:
            if other_experience["id"] == experience_id:
                continue
            
            # Check if experiences can be fused
            if self._can_fuse_experiences(experience, other_experience):
                fusion_candidates.append(other_experience)
        
        # Create fused experiences
        for candidate in fusion_candidates:
            fused_id = f"fused_{experience_id}_{candidate['id']}"
            fused_experience = await self._create_fused_experience(
                experience,
                candidate,
                fused_id
            )
            
            if fused_experience:
                self.fused_experiences[fused_id] = fused_experience
                experience["metadata"]["fusion_data"][fused_id] = {
                    "fused_with": candidate["id"],
                    "confidence": fused_experience["confidence"],
                    "timestamp": datetime.now().isoformat()
                }
        
        self.last_fusion = datetime.now()

    def _can_fuse_experiences(
        self,
        experience1: Dict[str, Any],
        experience2: Dict[str, Any]
    ) -> bool:
        """Check if two experiences can be fused."""
        # Check temporal proximity
        time1 = datetime.fromisoformat(experience1["timestamp"])
        time2 = datetime.fromisoformat(experience2["timestamp"])
        time_diff = abs((time1 - time2).total_seconds())
        
        if time_diff > 3600:  # 1 hour threshold
            return False
        
        # Check modality compatibility
        modalities1 = set(experience1["metadata"]["modalities"])
        modalities2 = set(experience2["metadata"]["modalities"])
        
        if not modalities1 or not modalities2:
            return False
        
        # Check location compatibility
        if (experience1["metadata"]["location"] and
            experience2["metadata"]["location"] and
            experience1["metadata"]["location"] != experience2["metadata"]["location"]):
            return False
        
        return True

    async def _create_fused_experience(
        self,
        experience1: Dict[str, Any],
        experience2: Dict[str, Any],
        fused_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create a fused experience from two experiences."""
        try:
            # Generate fusion prompt
            prompt = f"""
            Create a fused sensory experience from these two experiences:
            
            Experience 1: {experience1['content']}
            Modalities: {', '.join(experience1['metadata']['modalities'])}
            Intensity: {experience1['metadata']['intensity']}
            Valence: {experience1['metadata']['valence']}
            Arousal: {experience1['metadata']['arousal']}
            Location: {experience1['metadata']['location']}
            Context: {experience1['metadata']['context']}
            
            Experience 2: {experience2['content']}
            Modalities: {', '.join(experience2['metadata']['modalities'])}
            Intensity: {experience2['metadata']['intensity']}
            Valence: {experience2['metadata']['valence']}
            Arousal: {experience2['metadata']['arousal']}
            Location: {experience2['metadata']['location']}
            Context: {experience2['metadata']['context']}
            
            Return a JSON object with:
            1. content: string (fused description)
            2. modalities: list of strings
            3. intensity: float (0-1)
            4. valence: float (-1 to 1)
            5. arousal: float (0-1)
            6. confidence: float (0-1)
            7. fusion_type: string
            8. fusion_reason: string
            """
            response = await self.llm.generate(prompt)
            fusion = json.loads(response)
            
            return {
                "id": fused_id,
                "content": fusion["content"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "type": "fused",
                    "modalities": fusion["modalities"],
                    "intensity": fusion["intensity"],
                    "valence": fusion["valence"],
                    "arousal": fusion["arousal"],
                    "confidence": fusion["confidence"],
                    "fusion_type": fusion["fusion_type"],
                    "fusion_reason": fusion["fusion_reason"],
                    "source_experiences": [experience1["id"], experience2["id"]]
                }
            }
            
        except Exception as e:
            print(f"Error creating fused experience: {e}")
            return None

    async def _update_advanced_patterns(self) -> None:
        """Update advanced patterns in sensory experiences."""
        # Clear existing advanced patterns
        self.advanced_patterns = {}
        
        # Find temporal patterns
        temporal_patterns = self._find_temporal_patterns()
        
        # Find cross-modal patterns
        cross_modal_patterns = self._find_cross_modal_patterns()
        
        # Find fusion patterns
        fusion_patterns = self._find_fusion_patterns()
        
        # Combine patterns
        self.advanced_patterns = {
            **temporal_patterns,
            **cross_modal_patterns,
            **fusion_patterns
        }
        
        # Update experience metadata with pattern membership
        for pattern_id, pattern_data in self.advanced_patterns.items():
            for experience_id in pattern_data["experiences"]:
                experience = next(e for e in self.experiences if e["id"] == experience_id)
                experience["metadata"]["pattern_membership"].append(pattern_id)
        
        self.last_advanced_pattern = datetime.now()

    def _find_temporal_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Find temporal patterns in experiences."""
        patterns = {}
        
        # Sort experiences by timestamp
        sorted_experiences = sorted(
            self.experiences,
            key=lambda x: datetime.fromisoformat(x["timestamp"])
        )
        
        # Find sequences of related experiences
        current_sequence = []
        for i, experience in enumerate(sorted_experiences):
            if not current_sequence:
                current_sequence = [experience]
                continue
            
            # Check if experience belongs to current sequence
            if self._is_sequence_related(current_sequence, experience):
                current_sequence.append(experience)
            else:
                # Save sequence if it's long enough
                if len(current_sequence) >= 2:
                    pattern_id = f"temporal_pattern_{len(patterns)}"
                    patterns[pattern_id] = {
                        "type": "temporal",
                        "experiences": [e["id"] for e in current_sequence],
                        "start_time": current_sequence[0]["timestamp"],
                        "end_time": current_sequence[-1]["timestamp"],
                        "modalities": list(set(
                            modality
                            for e in current_sequence
                            for modality in e["metadata"]["modalities"]
                        ))
                    }
                current_sequence = [experience]
        
        return patterns

    def _is_sequence_related(
        self,
        sequence: List[Dict[str, Any]],
        experience: Dict[str, Any]
    ) -> bool:
        """Check if an experience is related to a sequence."""
        # Check temporal proximity
        last_time = datetime.fromisoformat(sequence[-1]["timestamp"])
        current_time = datetime.fromisoformat(experience["timestamp"])
        time_diff = abs((current_time - last_time).total_seconds())
        
        if time_diff > 3600:  # 1 hour threshold
            return False
        
        # Check modality overlap
        sequence_modalities = set(
            modality
            for e in sequence
            for modality in e["metadata"]["modalities"]
        )
        experience_modalities = set(experience["metadata"]["modalities"])
        
        if not sequence_modalities & experience_modalities:
            return False
        
        # Check location consistency
        if (sequence[-1]["metadata"]["location"] and
            experience["metadata"]["location"] and
            sequence[-1]["metadata"]["location"] != experience["metadata"]["location"]):
            return False
        
        return True

    def _find_cross_modal_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Find cross-modal patterns in experiences."""
        patterns = {}
        
        # Group experiences by location and context
        location_groups = {}
        for experience in self.experiences:
            location = experience["metadata"]["location"]
            if location:
                if location not in location_groups:
                    location_groups[location] = []
                location_groups[location].append(experience)
        
        # Find patterns in each location group
        for location, experiences in location_groups.items():
            # Find modality combinations
            modality_combinations = {}
            for experience in experiences:
                modalities = tuple(sorted(experience["metadata"]["modalities"]))
                if modalities not in modality_combinations:
                    modality_combinations[modalities] = []
                modality_combinations[modalities].append(experience)
            
            # Create patterns for significant combinations
            for modalities, group in modality_combinations.items():
                if len(group) >= 2:
                    pattern_id = f"cross_modal_pattern_{len(patterns)}"
                    patterns[pattern_id] = {
                        "type": "cross_modal",
                        "experiences": [e["id"] for e in group],
                        "modalities": list(modalities),
                        "location": location,
                        "frequency": len(group)
                    }
        
        return patterns

    def _find_fusion_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Find patterns in fused experiences."""
        patterns = {}
        
        # Group fused experiences by fusion type
        fusion_groups = {}
        for fused_id, fused in self.fused_experiences.items():
            fusion_type = fused["metadata"]["fusion_type"]
            if fusion_type not in fusion_groups:
                fusion_groups[fusion_type] = []
            fusion_groups[fusion_type].append(fused)
        
        # Create patterns for each fusion type
        for fusion_type, group in fusion_groups.items():
            if len(group) >= 2:
                pattern_id = f"fusion_pattern_{len(patterns)}"
                patterns[pattern_id] = {
                    "type": "fusion",
                    "experiences": [e["id"] for e in group],
                    "fusion_type": fusion_type,
                    "frequency": len(group),
                    "average_confidence": sum(
                        e["metadata"]["confidence"]
                        for e in group
                    ) / len(group)
                }
        
        return patterns 