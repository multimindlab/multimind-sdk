"""
Prospective Memory implementation for tracking future intentions and planned actions.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import networkx as nx
from .base import BaseMemory
from .temporal import TemporalMemory
from .semantic import SemanticMemory

class ProspectiveMemory(BaseMemory):
    """Memory implementation for future intentions and planned actions."""

    def __init__(
        self,
        reminder_threshold: timedelta = timedelta(hours=24),
        priority_levels: int = 5,
        max_intentions: int = 1000,
        **kwargs
    ):
        """Initialize prospective memory."""
        super().__init__(**kwargs)
        self.reminder_threshold = reminder_threshold
        self.priority_levels = priority_levels
        self.max_intentions = max_intentions
        
        # Component memories
        self.temporal_memory = TemporalMemory()
        self.semantic_memory = SemanticMemory()
        
        # Intention tracking
        self.intentions: Dict[str, Dict[str, Any]] = {}
        self.intention_graph = nx.DiGraph()
        
        # Reminders
        self.reminders: Dict[str, List[Dict[str, Any]]] = {}
        self.reminder_queue: List[Tuple[datetime, str]] = []

    async def add_intention(
        self,
        intention_id: str,
        description: str,
        planned_time: datetime,
        priority: int = 3,
        context: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        reminder_times: Optional[List[datetime]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a future intention with temporal and contextual information."""
        # Create intention entry
        intention = {
            'id': intention_id,
            'description': description,
            'planned_time': planned_time,
            'priority': min(priority, self.priority_levels),
            'context': context or {},
            'dependencies': dependencies or [],
            'status': 'pending',
            'created_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Store intention
        self.intentions[intention_id] = intention
        
        # Add to component memories
        await self.temporal_memory.add(intention_id, planned_time, metadata)
        await self.semantic_memory.add(intention_id, description, metadata)
        
        # Add to intention graph
        self.intention_graph.add_node(intention_id, **intention)
        
        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.intentions:
                    self.intention_graph.add_edge(dep_id, intention_id)
        
        # Set up reminders
        if reminder_times:
            self.reminders[intention_id] = []
            for reminder_time in reminder_times:
                reminder = {
                    'time': reminder_time,
                    'status': 'pending',
                    'created_at': datetime.now()
                }
                self.reminders[intention_id].append(reminder)
                self.reminder_queue.append((reminder_time, intention_id))
        
        # Sort reminder queue
        self.reminder_queue.sort(key=lambda x: x[0])

    async def get_intention(self, intention_id: str) -> Optional[Dict[str, Any]]:
        """Get an intention by ID."""
        return self.intentions.get(intention_id)

    async def get_pending_intentions(
        self,
        min_priority: Optional[int] = None,
        max_priority: Optional[int] = None,
        include_dependencies: bool = True
    ) -> List[Dict[str, Any]]:
        """Get pending intentions with optional priority filtering."""
        intentions = []
        for intention_id, intention in self.intentions.items():
            if intention['status'] == 'pending':
                if (min_priority is None or intention['priority'] >= min_priority) and \
                   (max_priority is None or intention['priority'] <= max_priority):
                    if include_dependencies:
                        intention['dependencies'] = list(self.intention_graph.predecessors(intention_id))
                    intentions.append(intention)
        return intentions

    async def get_upcoming_reminders(
        self,
        time_window: timedelta = timedelta(hours=24)
    ) -> List[Dict[str, Any]]:
        """Get reminders within a time window."""
        now = datetime.now()
        end_time = now + time_window
        
        reminders = []
        for reminder_time, intention_id in self.reminder_queue:
            if now <= reminder_time <= end_time:
                intention = self.intentions[intention_id]
                for reminder in self.reminders[intention_id]:
                    if reminder['time'] == reminder_time and reminder['status'] == 'pending':
                        reminders.append({
                            'intention': intention,
                            'reminder': reminder
                        })
        return reminders

    async def get_dependent_intentions(
        self,
        intention_id: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Get intentions that depend on a specific intention."""
        if intention_id not in self.intention_graph:
            return []
            
        dependent = []
        for node in nx.descendants_at_distance(self.intention_graph, intention_id, max_depth):
            dependent.append(self.intentions[node])
        return dependent

    async def update_intention(
        self,
        intention_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing intention."""
        if intention_id in self.intentions:
            intention = self.intentions[intention_id]
            intention.update(updates)
            
            # Update component memories
            if 'description' in updates:
                await self.semantic_memory.add(intention_id, updates['description'], intention['metadata'])
            if 'planned_time' in updates:
                await self.temporal_memory.add(intention_id, updates['planned_time'], intention['metadata'])
            
            # Update graph
            self.intention_graph.nodes[intention_id].update(updates)

    async def mark_reminder_complete(
        self,
        intention_id: str,
        reminder_time: datetime
    ) -> None:
        """Mark a reminder as complete."""
        if intention_id in self.reminders:
            for reminder in self.reminders[intention_id]:
                if reminder['time'] == reminder_time:
                    reminder['status'] = 'complete'
                    reminder['completed_at'] = datetime.now()

    async def mark_intention_complete(
        self,
        intention_id: str,
        completion_time: Optional[datetime] = None
    ) -> None:
        """Mark an intention as complete."""
        if intention_id in self.intentions:
            intention = self.intentions[intention_id]
            intention['status'] = 'complete'
            intention['completed_at'] = completion_time or datetime.now()
            
            # Update graph
            self.intention_graph.nodes[intention_id]['status'] = 'complete'

    async def remove_intention(self, intention_id: str) -> None:
        """Remove an intention."""
        if intention_id in self.intentions:
            # Remove from component memories
            await self.temporal_memory.remove(intention_id)
            await self.semantic_memory.remove(intention_id)
            
            # Remove from graph
            self.intention_graph.remove_node(intention_id)
            
            # Remove reminders
            if intention_id in self.reminders:
                del self.reminders[intention_id]
            
            # Remove from reminder queue
            self.reminder_queue = [
                (time, iid) for time, iid in self.reminder_queue
                if iid != intention_id
            ]
            
            # Remove intention
            del self.intentions[intention_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_intentions': len(self.intentions),
            'pending_intentions': len([
                i for i in self.intentions.values()
                if i['status'] == 'pending'
            ]),
            'completed_intentions': len([
                i for i in self.intentions.values()
                if i['status'] == 'complete'
            ]),
            'total_reminders': sum(len(r) for r in self.reminders.values()),
            'pending_reminders': sum(
                len([r for r in reminders if r['status'] == 'pending'])
                for reminders in self.reminders.values()
            ),
            'intention_graph_size': self.intention_graph.number_of_nodes(),
            'intention_graph_edges': self.intention_graph.number_of_edges()
        } 