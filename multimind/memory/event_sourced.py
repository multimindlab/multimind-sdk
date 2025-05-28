"""
Event-sourced memory implementation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class EventSourcedMemory(BaseMemory):
    """Memory that implements event-sourced memory."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_items: int = 1000,
        max_events: int = 10000,
        event_retention_days: int = 30,
        enable_event_analysis: bool = True,
        analysis_interval: int = 3600,  # 1 hour
        enable_pattern_detection: bool = True,
        pattern_threshold: float = 0.7,
        enable_causality_analysis: bool = True,
        causality_threshold: float = 0.6,
        enable_optimization: bool = True,
        optimization_interval: int = 3600  # 1 hour
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_items = max_items
        self.max_events = max_events
        self.event_retention_days = event_retention_days
        self.enable_event_analysis = enable_event_analysis
        self.analysis_interval = analysis_interval
        self.enable_pattern_detection = enable_pattern_detection
        self.pattern_threshold = pattern_threshold
        self.enable_causality_analysis = enable_causality_analysis
        self.causality_threshold = causality_threshold
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        
        # Initialize storage
        self.items: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []  # Event log
        self.patterns: Dict[str, List[Dict[str, Any]]] = {}  # pattern_id -> pattern data
        self.causal_chains: Dict[str, List[Dict[str, Any]]] = {}  # chain_id -> causal chain
        self.last_analysis = datetime.now()
        self.last_optimization = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message and create events."""
        # Create new item
        item_id = f"item_{len(self.items)}"
        new_item = {
            "id": item_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "message",
                "created_at": datetime.now().isoformat(),
                "modified_at": datetime.now().isoformat(),
                "event_count": 0,
                "pattern_count": 0,
                "causal_count": 0
            }
        }
        
        # Add to storage
        self.items.append(new_item)
        
        # Create events
        await self._create_events(item_id, new_item)
        
        # Analyze events if needed
        if self.enable_event_analysis and (
            datetime.now() - self.last_analysis
        ).total_seconds() >= self.analysis_interval:
            await self._analyze_events()
        
        # Maintain item limit
        await self._maintain_item_limit()
        
        await self.save()

    async def _create_events(self, item_id: str, item: Dict[str, Any]) -> None:
        """Create events for a new item."""
        # Create creation event
        creation_event = {
            "id": f"event_{len(self.events)}",
            "type": "item_created",
            "timestamp": datetime.now().isoformat(),
            "item_id": item_id,
            "data": {
                "content": item["content"],
                "metadata": item["metadata"]
            }
        }
        self.events.append(creation_event)
        
        # Create analysis events
        if self.enable_pattern_detection:
            await self._create_pattern_events(item_id, item)
        
        if self.enable_causality_analysis:
            await self._create_causality_events(item_id, item)
        
        # Update item metadata
        item["metadata"]["event_count"] = len([
            e for e in self.events if e["item_id"] == item_id
        ])

    async def _create_pattern_events(self, item_id: str, item: Dict[str, Any]) -> None:
        """Create pattern detection events."""
        try:
            # Generate pattern analysis prompt
            prompt = f"""
            Analyze patterns in this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. patterns: list of strings
            2. pattern_types: list of strings
            3. pattern_confidence: list of floats
            """
            response = await self.llm.generate(prompt)
            patterns = json.loads(response)
            
            # Create pattern events
            for i, pattern in enumerate(patterns["patterns"]):
                pattern_event = {
                    "id": f"event_{len(self.events)}",
                    "type": "pattern_detected",
                    "timestamp": datetime.now().isoformat(),
                    "item_id": item_id,
                    "data": {
                        "pattern": pattern,
                        "pattern_type": patterns["pattern_types"][i],
                        "confidence": patterns["pattern_confidence"][i]
                    }
                }
                self.events.append(pattern_event)
                
                # Update patterns
                pattern_id = f"pattern_{len(self.patterns)}"
                if pattern_id not in self.patterns:
                    self.patterns[pattern_id] = []
                self.patterns[pattern_id].append({
                    "item_id": item_id,
                    "event_id": pattern_event["id"],
                    "timestamp": pattern_event["timestamp"]
                })
            
            # Update item metadata
            item["metadata"]["pattern_count"] = len(patterns["patterns"])
            
        except Exception as e:
            print(f"Error creating pattern events: {e}")

    async def _create_causality_events(self, item_id: str, item: Dict[str, Any]) -> None:
        """Create causality analysis events."""
        try:
            # Generate causality analysis prompt
            prompt = f"""
            Analyze causality for this item:
            
            {item['content']}
            
            Return a JSON object with:
            1. causes: list of strings
            2. effects: list of strings
            3. confidence: list of floats
            """
            response = await self.llm.generate(prompt)
            causality = json.loads(response)
            
            # Create causality events
            for i, cause in enumerate(causality["causes"]):
                causality_event = {
                    "id": f"event_{len(self.events)}",
                    "type": "causality_detected",
                    "timestamp": datetime.now().isoformat(),
                    "item_id": item_id,
                    "data": {
                        "cause": cause,
                        "effect": causality["effects"][i],
                        "confidence": causality["confidence"][i]
                    }
                }
                self.events.append(causality_event)
                
                # Update causal chains
                chain_id = f"chain_{len(self.causal_chains)}"
                if chain_id not in self.causal_chains:
                    self.causal_chains[chain_id] = []
                self.causal_chains[chain_id].append({
                    "item_id": item_id,
                    "event_id": causality_event["id"],
                    "timestamp": causality_event["timestamp"]
                })
            
            # Update item metadata
            item["metadata"]["causal_count"] = len(causality["causes"])
            
        except Exception as e:
            print(f"Error creating causality events: {e}")

    async def _analyze_events(self) -> None:
        """Analyze event patterns and causality."""
        # Analyze event patterns
        if self.enable_pattern_detection:
            await self._analyze_patterns()
        
        # Analyze causality
        if self.enable_causality_analysis:
            await self._analyze_causality()
        
        # Update last analysis time
        self.last_analysis = datetime.now()

    async def _analyze_patterns(self) -> None:
        """Analyze event patterns."""
        # Group events by type
        event_groups = {}
        for event in self.events:
            if event["type"] not in event_groups:
                event_groups[event["type"]] = []
            event_groups[event["type"]].append(event)
        
        # Analyze each group
        for event_type, events in event_groups.items():
            try:
                # Generate pattern analysis prompt
                prompt = f"""
                Analyze patterns in these events:
                
                {json.dumps(events, indent=2)}
                
                Return a JSON object with:
                1. patterns: list of strings
                2. pattern_types: list of strings
                3. pattern_confidence: list of floats
                """
                response = await self.llm.generate(prompt)
                patterns = json.loads(response)
                
                # Update patterns
                for i, pattern in enumerate(patterns["patterns"]):
                    pattern_id = f"pattern_{len(self.patterns)}"
                    self.patterns[pattern_id] = [
                        {
                            "event_id": event["id"],
                            "timestamp": event["timestamp"]
                        }
                        for event in events
                    ]
                
            except Exception as e:
                print(f"Error analyzing patterns: {e}")

    async def _analyze_causality(self) -> None:
        """Analyze event causality."""
        # Group events by item
        item_events = {}
        for event in self.events:
            if event["item_id"] not in item_events:
                item_events[event["item_id"]] = []
            item_events[event["item_id"]].append(event)
        
        # Analyze each item's events
        for item_id, events in item_events.items():
            try:
                # Generate causality analysis prompt
                prompt = f"""
                Analyze causality in these events:
                
                {json.dumps(events, indent=2)}
                
                Return a JSON object with:
                1. causes: list of strings
                2. effects: list of strings
                3. confidence: list of floats
                """
                response = await self.llm.generate(prompt)
                causality = json.loads(response)
                
                # Update causal chains
                for i, cause in enumerate(causality["causes"]):
                    chain_id = f"chain_{len(self.causal_chains)}"
                    self.causal_chains[chain_id] = [
                        {
                            "event_id": event["id"],
                            "timestamp": event["timestamp"]
                        }
                        for event in events
                    ]
                
            except Exception as e:
                print(f"Error analyzing causality: {e}")

    async def _maintain_item_limit(self) -> None:
        """Maintain item and event limits."""
        # Check item limit
        if len(self.items) > self.max_items:
            # Sort items by timestamp
            sorted_items = sorted(
                self.items,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest items
            items_to_remove = sorted_items[:len(self.items) - self.max_items]
            for item in items_to_remove:
                await self._remove_item(item["id"])
        
        # Check event limit
        if len(self.events) > self.max_events:
            # Sort events by timestamp
            sorted_events = sorted(
                self.events,
                key=lambda x: datetime.fromisoformat(x["timestamp"])
            )
            
            # Remove oldest events
            self.events = sorted_events[len(self.events) - self.max_events:]

    async def _remove_item(self, item_id: str) -> None:
        """Remove an item and its associated events."""
        # Remove from items
        self.items = [i for i in self.items if i["id"] != item_id]
        
        # Remove associated events
        self.events = [e for e in self.events if e["item_id"] != item_id]
        
        # Remove from patterns
        for pattern_id, pattern_data in self.patterns.items():
            self.patterns[pattern_id] = [
                p for p in pattern_data if p["item_id"] != item_id
            ]
        
        # Remove from causal chains
        for chain_id, chain_data in self.causal_chains.items():
            self.causal_chains[chain_id] = [
                c for c in chain_data if c["item_id"] != item_id
            ]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all items."""
        messages = []
        for item in self.items:
            messages.append({
                "role": "event_sourced_memory",
                "content": item["content"],
                "timestamp": item["timestamp"]
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all items and events."""
        self.items = []
        self.events = []
        self.patterns = {}
        self.causal_chains = {}
        await self.save()

    async def save(self) -> None:
        """Save items and events to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "items": self.items,
                    "events": self.events,
                    "patterns": self.patterns,
                    "causal_chains": self.causal_chains,
                    "last_analysis": self.last_analysis.isoformat(),
                    "last_optimization": self.last_optimization.isoformat()
                }, f)

    def load(self) -> None:
        """Load items and events from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.items = data.get("items", [])
                self.events = data.get("events", [])
                self.patterns = data.get("patterns", {})
                self.causal_chains = data.get("causal_chains", {})
                self.last_analysis = datetime.fromisoformat(
                    data.get("last_analysis", datetime.now().isoformat())
                )
                self.last_optimization = datetime.fromisoformat(
                    data.get("last_optimization", datetime.now().isoformat())
                )

    async def get_event_sourced_stats(self) -> Dict[str, Any]:
        """Get statistics about event-sourced memory."""
        stats = {
            "total_items": len(self.items),
            "event_stats": {
                "total_events": len(self.events),
                "event_types": len(set(e["type"] for e in self.events)),
                "average_events_per_item": len(self.events) / len(self.items) if self.items else 0
            },
            "pattern_stats": {
                "total_patterns": len(self.patterns),
                "average_patterns_per_item": sum(
                    len(patterns) for patterns in self.patterns.values()
                ) / len(self.patterns) if self.patterns else 0
            },
            "causality_stats": {
                "total_chains": len(self.causal_chains),
                "average_chain_length": sum(
                    len(chain) for chain in self.causal_chains.values()
                ) / len(self.causal_chains) if self.causal_chains else 0
            }
        }
        
        return stats

    async def get_event_sourced_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for event-sourced memory optimization."""
        suggestions = []
        
        # Check item count
        if len(self.items) > self.max_items * 0.8:
            suggestions.append({
                "type": "item_limit",
                "suggestion": "Consider increasing max_items or removing older items"
            })
        
        # Check event count
        stats = await self.get_event_sourced_stats()
        if stats["event_stats"]["total_events"] > self.max_events * 0.8:
            suggestions.append({
                "type": "event_limit",
                "suggestion": "Consider increasing max_events or compressing events"
            })
        
        # Check pattern coverage
        if stats["pattern_stats"]["average_patterns_per_item"] < 2:
            suggestions.append({
                "type": "pattern_coverage",
                "suggestion": "Consider improving pattern detection"
            })
        
        # Check causality coverage
        if stats["causality_stats"]["average_chain_length"] < 2:
            suggestions.append({
                "type": "causality_coverage",
                "suggestion": "Consider improving causality analysis"
            })
        
        return suggestions 