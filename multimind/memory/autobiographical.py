"""
Autobiographical Memory implementation for tracking personal experiences and life events.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
import networkx as nx
from .base import BaseMemory
from .episodic import EpisodicMemory
from .emotional import EmotionalMemory
from .temporal import TemporalMemory

class AutobiographicalMemory(BaseMemory):
    """Memory implementation for personal experiences and life events."""

    def __init__(
        self,
        emotional_threshold: float = 0.5,
        temporal_decay: float = 0.95,
        max_events: int = 1000,
        **kwargs
    ):
        """Initialize autobiographical memory."""
        super().__init__(**kwargs)
        self.emotional_threshold = emotional_threshold
        self.temporal_decay = temporal_decay
        self.max_events = max_events
        
        # Component memories
        self.episodic_memory = EpisodicMemory()
        self.emotional_memory = EmotionalMemory()
        self.temporal_memory = TemporalMemory()
        
        # Event tracking
        self.events: Dict[str, Dict[str, Any]] = {}
        self.event_graph = nx.DiGraph()
        
        # Life periods
        self.life_periods: Dict[str, Dict[str, Any]] = {}
        self.current_period: Optional[str] = None

    async def add_event(
        self,
        event_id: str,
        description: str,
        timestamp: datetime,
        location: Optional[str] = None,
        participants: Optional[List[str]] = None,
        emotional_valence: Optional[float] = None,
        emotional_arousal: Optional[float] = None,
        period: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a life event with emotional and temporal context."""
        # Create event entry
        event = {
            'id': event_id,
            'description': description,
            'timestamp': timestamp,
            'location': location,
            'participants': participants or [],
            'emotional_valence': emotional_valence,
            'emotional_arousal': emotional_arousal,
            'period': period,
            'metadata': metadata or {},
            'created_at': datetime.now()
        }
        
        # Store event
        self.events[event_id] = event
        
        # Add to component memories
        await self.episodic_memory.add(event_id, description, metadata)
        if emotional_valence is not None and emotional_arousal is not None:
            await self.emotional_memory.add(
                event_id,
                {'valence': emotional_valence, 'arousal': emotional_arousal},
                metadata
            )
        await self.temporal_memory.add(event_id, timestamp, metadata)
        
        # Add to event graph
        self.event_graph.add_node(event_id, **event)
        
        # Link to life period
        if period:
            if period not in self.life_periods:
                self.life_periods[period] = {
                    'start_time': timestamp,
                    'end_time': None,
                    'events': []
                }
            self.life_periods[period]['events'].append(event_id)
            self.current_period = period

    async def add_life_period(
        self,
        period_id: str,
        start_time: datetime,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new life period."""
        self.life_periods[period_id] = {
            'start_time': start_time,
            'end_time': None,
            'description': description,
            'metadata': metadata or {},
            'events': []
        }
        self.current_period = period_id

    async def end_life_period(self, period_id: str, end_time: datetime) -> None:
        """End a life period."""
        if period_id in self.life_periods:
            self.life_periods[period_id]['end_time'] = end_time

    async def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a life event by ID."""
        return self.events.get(event_id)

    async def get_events_by_period(
        self,
        period_id: str,
        include_emotional: bool = True,
        include_temporal: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all events in a life period."""
        if period_id not in self.life_periods:
            return []
            
        events = []
        for event_id in self.life_periods[period_id]['events']:
            event = self.events[event_id]
            
            if include_emotional:
                emotional = await self.emotional_memory.get(event_id)
                if emotional:
                    event['emotional'] = emotional
                    
            if include_temporal:
                temporal = await self.temporal_memory.get(event_id)
                if temporal:
                    event['temporal'] = temporal
                    
            events.append(event)
            
        return events

    async def get_emotional_events(
        self,
        min_valence: Optional[float] = None,
        max_valence: Optional[float] = None,
        min_arousal: Optional[float] = None,
        max_arousal: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get events with specific emotional characteristics."""
        events = []
        for event_id, event in self.events.items():
            if event['emotional_valence'] is not None and event['emotional_arousal'] is not None:
                if (min_valence is None or event['emotional_valence'] >= min_valence) and \
                   (max_valence is None or event['emotional_valence'] <= max_valence) and \
                   (min_arousal is None or event['emotional_arousal'] >= min_arousal) and \
                   (max_arousal is None or event['emotional_arousal'] <= max_arousal):
                    events.append(event)
        return events

    async def get_temporal_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get events within a time range."""
        events = []
        for event_id, event in self.events.items():
            if (start_time is None or event['timestamp'] >= start_time) and \
               (end_time is None or event['timestamp'] <= end_time):
                events.append(event)
        return events

    async def get_related_events(
        self,
        event_id: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get events related to a specific event through the event graph."""
        if event_id not in self.event_graph:
            return []
            
        related = []
        for node in nx.descendants_at_distance(self.event_graph, event_id, max_depth):
            related.append(self.events[node])
        return related

    async def update_event(
        self,
        event_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing event."""
        if event_id in self.events:
            event = self.events[event_id]
            event.update(updates)
            
            # Update component memories
            if 'description' in updates:
                await self.episodic_memory.add(event_id, updates['description'], event['metadata'])
            if 'emotional_valence' in updates or 'emotional_arousal' in updates:
                await self.emotional_memory.add(
                    event_id,
                    {
                        'valence': event['emotional_valence'],
                        'arousal': event['emotional_arousal']
                    },
                    event['metadata']
                )
            if 'timestamp' in updates:
                await self.temporal_memory.add(event_id, updates['timestamp'], event['metadata'])

    async def remove_event(self, event_id: str) -> None:
        """Remove a life event."""
        if event_id in self.events:
            event = self.events[event_id]
            
            # Remove from component memories
            await self.episodic_memory.remove(event_id)
            await self.emotional_memory.remove(event_id)
            await self.temporal_memory.remove(event_id)
            
            # Remove from life period
            if event['period'] in self.life_periods:
                self.life_periods[event['period']]['events'].remove(event_id)
            
            # Remove from event graph
            self.event_graph.remove_node(event_id)
            
            # Remove from events
            del self.events[event_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_events': len(self.events),
            'total_periods': len(self.life_periods),
            'current_period': self.current_period,
            'emotional_events': len([
                e for e in self.events.values()
                if e['emotional_valence'] is not None and e['emotional_arousal'] is not None
            ]),
            'temporal_events': len([
                e for e in self.events.values()
                if e['timestamp'] is not None
            ]),
            'event_graph_size': self.event_graph.number_of_nodes(),
            'event_graph_edges': self.event_graph.number_of_edges()
        } 