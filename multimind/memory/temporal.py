"""
Temporal memory implementation that manages time-based information and temporal relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class TemporalMemory(BaseMemory):
    """Memory that manages time-based information and temporal relationships."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_events: int = 1000,
        temporal_threshold: float = 0.7,
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
        relationship_types: Set[str] = None
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_events = max_events
        self.temporal_threshold = temporal_threshold
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
        self.relationship_types = relationship_types or {
            "before",
            "after",
            "during",
            "overlaps",
            "starts",
            "ends",
            "contains",
            "contained_by",
            "concurrent",
            "precedes",
            "follows"
        }
        
        # Initialize temporal memory storage
        self.events: List[Dict[str, Any]] = []
        self.event_embeddings: List[List[float]] = []
        self.relationships: Dict[str, Dict[str, List[str]]] = {}  # event_id -> {relationship_type -> target_ids}
        self.patterns: Dict[str, List[str]] = {}  # pattern_id -> event_ids
        self.learning_history: Dict[str, List[Dict[str, Any]]] = {}  # event_id -> learning records
        self.event_history: List[Dict[str, Any]] = []  # Recent event updates
        self.evolution_history: Dict[str, List[Dict[str, Any]]] = {}  # event_id -> evolution records
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}  # event_id -> validation records
        self.last_analysis = datetime.now()
        self.last_relationship_update = datetime.now()
        self.last_pattern_update = datetime.now()
        self.last_evolution = datetime.now()
        self.last_validation = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and analyze temporal information."""
        # Create new event
        event_id = f"event_{len(self.events)}"
        new_event = {
            "id": event_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "start_time": None,
                "end_time": None,
                "duration": None,
                "temporal_type": None,
                "importance": 0.0,
                "recurrence": None,
                "learning_progress": 0.0,
                "evolution_stage": 0,
                "validation_score": 0.0,
                "analysis_results": {},
                "validation_results": {}
            }
        }
        
        # Add to storage
        self.events.append(new_event)
        
        # Get event embedding
        embedding = await self.llm.embeddings(message["content"])
        self.event_embeddings.append(embedding)
        
        # Analyze temporal information
        if self.enable_analysis:
            current_time = datetime.now()
            if (current_time - self.last_analysis).total_seconds() > self.analysis_interval:
                await self._analyze_temporal_info(event_id)
        
        # Find relationships
        if self.enable_relationships:
            current_time = datetime.now()
            if (current_time - self.last_relationship_update).total_seconds() > self.relationship_interval:
                await self._find_relationships(event_id)
        
        # Update patterns
        if self.enable_patterns:
            current_time = datetime.now()
            if (current_time - self.last_pattern_update).total_seconds() > self.pattern_interval:
                await self._update_patterns()
        
        # Update event history
        if self.enable_history:
            self.event_history.append({
                "event_id": event_id,
                "timestamp": new_event["timestamp"],
                "content": new_event["content"],
                "start_time": new_event["metadata"]["start_time"],
                "end_time": new_event["metadata"]["end_time"],
                "temporal_type": new_event["metadata"]["temporal_type"]
            })
            if len(self.event_history) > self.history_window:
                self.event_history.pop(0)
        
        # Update learning progress
        if self.enable_learning:
            await self._update_learning_progress(event_id)
        
        # Update evolution
        if self.enable_evolution:
            current_time = datetime.now()
            if (current_time - self.last_evolution).total_seconds() > self.evolution_interval:
                await self._update_evolution(event_id)
        
        # Validate event
        if self.enable_validation:
            current_time = datetime.now()
            if (current_time - self.last_validation).total_seconds() > self.validation_interval:
                await self._validate_event(event_id)
        
        # Maintain event limit
        await self._maintain_event_limit()
        
        await self.save()

    async def _analyze_temporal_info(self, event_id: str) -> None:
        """Analyze temporal information from a message."""
        event = next(e for e in self.events if e["id"] == event_id)
        
        try:
            # Generate analysis prompt
            prompt = f"""
            Analyze the temporal information in this message:
            
            {event['content']}
            
            Return a JSON object with:
            1. start_time: string (ISO format) or null
            2. end_time: string (ISO format) or null
            3. duration: string (e.g., "2 hours") or null
            4. temporal_type: string (e.g., point, interval, recurring)
            5. importance: float (0-1)
            6. recurrence: dict with pattern info or null
            """
            response = await self.llm.generate(prompt)
            analysis = json.loads(response)
            
            # Update event metadata
            event["metadata"]["start_time"] = analysis.get("start_time")
            event["metadata"]["end_time"] = analysis.get("end_time")
            event["metadata"]["duration"] = analysis.get("duration")
            event["metadata"]["temporal_type"] = analysis.get("temporal_type")
            event["metadata"]["importance"] = analysis.get("importance", 0.0)
            event["metadata"]["recurrence"] = analysis.get("recurrence")
            event["metadata"]["analysis_results"] = analysis
            
        except Exception as e:
            print(f"Error analyzing temporal info: {e}")

    async def _find_relationships(self, event_id: str) -> None:
        """Find temporal relationships between events."""
        event = next(e for e in self.events if e["id"] == event_id)
        
        for other_event in self.events:
            if other_event["id"] == event_id:
                continue
            
            # Calculate temporal similarity
            similarity = self._calculate_temporal_similarity(
                event["metadata"],
                other_event["metadata"]
            )
            
            if similarity >= self.temporal_threshold:
                # Determine relationship type
                relationship_type = await self._determine_relationship_type(
                    event,
                    other_event,
                    similarity
                )
                
                if relationship_type:
                    # Add bidirectional relationship
                    self.relationships[event_id][relationship_type].append(other_event["id"])
                    self.relationships[other_event["id"]][relationship_type].append(event_id)

    def _calculate_temporal_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two temporal events."""
        # Calculate time similarity if available
        time_similarity = 0.0
        if metadata1["start_time"] and metadata2["start_time"]:
            time1 = datetime.fromisoformat(metadata1["start_time"])
            time2 = datetime.fromisoformat(metadata2["start_time"])
            time_diff = abs((time1 - time2).total_seconds())
            time_similarity = 1.0 / (1.0 + time_diff / 86400)  # Normalize by day
        
        # Calculate duration similarity if available
        duration_similarity = 0.0
        if metadata1["duration"] and metadata2["duration"]:
            # Simple duration comparison (could be enhanced)
            duration_similarity = 1.0 if metadata1["duration"] == metadata2["duration"] else 0.0
        
        # Calculate type similarity
        type_similarity = 1.0 if metadata1["temporal_type"] == metadata2["temporal_type"] else 0.0
        
        # Calculate importance similarity
        importance_similarity = 1.0 - abs(metadata1["importance"] - metadata2["importance"])
        
        return (time_similarity + duration_similarity + type_similarity + importance_similarity) / 4

    async def _determine_relationship_type(
        self,
        event1: Dict[str, Any],
        event2: Dict[str, Any],
        similarity: float
    ) -> Optional[str]:
        """Determine the type of temporal relationship between two events."""
        try:
            prompt = f"""
            Determine the temporal relationship type between these two events:
            
            Event 1: {event1['content']}
            Start Time: {event1['metadata']['start_time']}
            End Time: {event1['metadata']['end_time']}
            Duration: {event1['metadata']['duration']}
            Type: {event1['metadata']['temporal_type']}
            
            Event 2: {event2['content']}
            Start Time: {event2['metadata']['start_time']}
            End Time: {event2['metadata']['end_time']}
            Duration: {event2['metadata']['duration']}
            Type: {event2['metadata']['temporal_type']}
            
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
        """Update patterns of related events."""
        # Clear existing patterns
        self.patterns = {}
        
        # Group by relationship types
        for relationship_type in self.relationship_types:
            # Find connected components
            visited = set()
            
            for event_id in self.relationships:
                if event_id in visited:
                    continue
                
                # Start new pattern
                pattern_id = f"pattern_{len(self.patterns)}"
                pattern = []
                
                # DFS to find connected events
                stack = [event_id]
                while stack:
                    current_id = stack.pop()
                    if current_id in visited:
                        continue
                    
                    visited.add(current_id)
                    pattern.append(current_id)
                    
                    # Add related events
                    for related_id in self.relationships[current_id][relationship_type]:
                        if related_id not in visited:
                            stack.append(related_id)
                
                if len(pattern) >= 2:  # Minimum pattern size
                    self.patterns[pattern_id] = pattern
        
        self.last_pattern_update = datetime.now()

    async def _update_learning_progress(self, event_id: str) -> None:
        """Update learning progress for an event."""
        event = next(e for e in self.events if e["id"] == event_id)
        
        # Calculate learning metrics
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[event_id].values()
        )
        importance = event["metadata"]["importance"]
        validation_score = event["metadata"]["validation_score"]
        
        # Update learning progress
        progress = (
            self.learning_rate * (relationship_count / len(self.relationship_types)) +
            self.learning_rate * importance +
            self.learning_rate * validation_score
        )
        
        event["metadata"]["learning_progress"] = min(
            1.0,
            event["metadata"]["learning_progress"] + progress
        )
        
        # Record learning update
        self.learning_history[event_id].append({
            "timestamp": datetime.now().isoformat(),
            "relationship_count": relationship_count,
            "importance": importance,
            "validation_score": validation_score,
            "progress": progress
        })

    async def _update_evolution(self, event_id: str) -> None:
        """Update evolution stage for an event."""
        event = next(e for e in self.events if e["id"] == event_id)
        
        # Calculate evolution metrics
        learning_progress = event["metadata"]["learning_progress"]
        relationship_count = sum(
            len(relationships)
            for relationships in self.relationships[event_id].values()
        )
        validation_score = event["metadata"]["validation_score"]
        
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
        event["metadata"]["evolution_stage"] = stage
        
        # Record evolution
        self.evolution_history[event_id].append({
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "learning_progress": learning_progress,
            "relationship_count": relationship_count,
            "validation_score": validation_score
        })

    async def _validate_event(self, event_id: str) -> None:
        """Validate temporal information of an event."""
        event = next(e for e in self.events if e["id"] == event_id)
        
        try:
            # Generate validation prompt
            prompt = f"""
            Validate the temporal information of this event:
            
            {event['content']}
            
            Start Time: {event['metadata']['start_time']}
            End Time: {event['metadata']['end_time']}
            Duration: {event['metadata']['duration']}
            Type: {event['metadata']['temporal_type']}
            
            Return a JSON object with:
            1. validation_score: float (0-1)
            2. validation_reason: string
            3. inconsistencies: list of strings
            4. suggestions: list of strings
            """
            response = await self.llm.generate(prompt)
            validation = json.loads(response)
            
            # Update event metadata
            event["metadata"]["validation_score"] = validation["validation_score"]
            event["metadata"]["validation_results"] = validation
            
            # Record validation
            self.validation_history[event_id].append({
                "timestamp": datetime.now().isoformat(),
                "score": validation["validation_score"],
                "reason": validation["validation_reason"],
                "inconsistencies": validation["inconsistencies"],
                "suggestions": validation["suggestions"]
            })
            
        except Exception as e:
            print(f"Error validating event: {e}")

    async def _maintain_event_limit(self) -> None:
        """Maintain event limit by removing least important events."""
        if len(self.events) > self.max_events:
            # Sort events by learning progress and validation score
            sorted_events = sorted(
                self.events,
                key=lambda x: (
                    x["metadata"]["learning_progress"] +
                    x["metadata"]["validation_score"]
                )
            )
            
            # Remove events with lowest scores
            events_to_remove = sorted_events[:len(self.events) - self.max_events]
            for event in events_to_remove:
                await self._remove_event(event["id"])

    async def _remove_event(self, event_id: str) -> None:
        """Remove an event and its associated data."""
        # Remove from events
        event_idx = next(i for i, e in enumerate(self.events) if e["id"] == event_id)
        self.events.pop(event_idx)
        self.event_embeddings.pop(event_idx)
        
        # Remove from relationships
        if event_id in self.relationships:
            del self.relationships[event_id]
        
        # Remove from patterns
        for pattern_id, pattern in self.patterns.items():
            if event_id in pattern:
                pattern.remove(event_id)
                if len(pattern) < 2:  # Minimum pattern size
                    del self.patterns[pattern_id]
        
        # Remove from history
        if self.enable_history:
            self.event_history = [
                e for e in self.event_history
                if e["event_id"] != event_id
            ]
        
        # Remove learning history
        if event_id in self.learning_history:
            del self.learning_history[event_id]
        
        # Remove evolution history
        if event_id in self.evolution_history:
            del self.evolution_history[event_id]
        
        # Remove validation history
        if event_id in self.validation_history:
            del self.validation_history[event_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all events."""
        messages = []
        for event in self.events:
            messages.append({
                "role": "temporal_memory",
                "content": event["content"],
                "timestamp": event["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all events."""
        self.events = []
        self.event_embeddings = []
        self.relationships = {}
        self.patterns = {}
        self.learning_history = {}
        self.event_history = []
        self.evolution_history = {}
        self.validation_history = {}
        await self.save()

    async def save(self) -> None:
        """Save events to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "events": self.events,
                    "relationships": self.relationships,
                    "patterns": self.patterns,
                    "learning_history": self.learning_history,
                    "event_history": self.event_history,
                    "evolution_history": self.evolution_history,
                    "validation_history": self.validation_history,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_relationship_update": self.last_relationship_update.isoformat(),
                    "last_pattern_update": self.last_pattern_update.isoformat(),
                    "last_evolution": self.last_evolution.isoformat(),
                    "last_validation": self.last_validation.isoformat()
                }, f)

    def load(self) -> None:
        """Load events from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.events = data.get("events", [])
                self.relationships = data.get("relationships", {})
                self.patterns = data.get("patterns", {})
                self.learning_history = data.get("learning_history", {})
                self.event_history = data.get("event_history", [])
                self.evolution_history = data.get("evolution_history", {})
                self.validation_history = data.get("validation_history", {})
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
                
                # Recreate embeddings
                self.event_embeddings = []
                for event in self.events:
                    self.event_embeddings.append(
                        self.llm.embeddings(event["content"])
                    )

    async def get_temporal_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about temporal memory."""
        stats = {
            "total_events": len(self.events),
            "temporal_type_distribution": {
                event_type: sum(1 for e in self.events if e["metadata"]["temporal_type"] == event_type)
                for event_type in set(e["metadata"]["temporal_type"] for e in self.events if e["metadata"]["temporal_type"])
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
                    for e in self.events
                ) / len(self.events) if self.events else 0,
                "events_with_progress": sum(
                    1 for e in self.events
                    if e["metadata"]["learning_progress"] > 0
                )
            },
            "evolution_stats": {
                "stage_distribution": {
                    stage: sum(1 for e in self.events if e["metadata"]["evolution_stage"] == stage)
                    for stage in range(4)
                },
                "average_stage": sum(e["metadata"]["evolution_stage"] for e in self.events) / len(self.events) if self.events else 0
            },
            "validation_stats": {
                "average_score": sum(
                    e["metadata"]["validation_score"]
                    for e in self.events
                ) / len(self.events) if self.events else 0,
                "validated_events": sum(
                    1 for e in self.events
                    if e["metadata"]["validation_score"] >= 0.8
                )
            }
        }
        
        return stats

    async def get_temporal_memory_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for temporal memory optimization."""
        suggestions = []
        
        # Check event count
        if len(self.events) > self.max_events * 0.8:
            suggestions.append({
                "type": "event_limit",
                "suggestion": "Consider increasing max_events or removing less important events"
            })
        
        # Check relationship quality
        stats = await self.get_temporal_memory_stats()
        if stats["relationship_stats"]["total_relationships"] < len(self.events) * 2:
            suggestions.append({
                "type": "relationship_development",
                "suggestion": "Consider developing more temporal relationships between events"
            })
        
        # Check pattern quality
        if stats["pattern_stats"]["average_pattern_size"] < 2:
            suggestions.append({
                "type": "pattern_development",
                "suggestion": "Consider developing more temporal patterns or adjusting pattern detection"
            })
        
        # Check learning progress
        if stats["learning_stats"]["average_progress"] < 0.5:
            suggestions.append({
                "type": "learning_enhancement",
                "suggestion": "Consider enhancing learning mechanisms for events"
            })
        
        # Check evolution progress
        if stats["evolution_stats"]["average_stage"] < 1.5:
            suggestions.append({
                "type": "evolution_enhancement",
                "suggestion": "Consider enhancing evolution mechanisms for events"
            })
        
        # Check validation quality
        if stats["validation_stats"]["average_score"] < 0.8:
            suggestions.append({
                "type": "validation_improvement",
                "suggestion": "Consider improving validation mechanisms or resolving inconsistencies"
            })
        
        return suggestions 