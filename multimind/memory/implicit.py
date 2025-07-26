from __future__ import annotations

"""
Implicit Memory implementation for storing unconscious, procedural knowledge.
"""

from datetime import timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime
import numpy as np
from .base import BaseMemory
from .procedural import ProceduralMemory
from .semantic import SemanticMemory

class ImplicitMemory(BaseMemory):
    """Memory implementation for unconscious, procedural knowledge."""

    def __init__(
        self,
        skill_decay: float = 0.95,
        max_skills: int = 1000,
        **kwargs
    ):
        """Initialize implicit memory."""
        super().__init__(**kwargs)
        self.skill_decay = skill_decay
        self.max_skills = max_skills
        
        # Component memories
        self.procedural_memory = ProceduralMemory()
        self.semantic_memory = SemanticMemory()
        
        # Skill tracking
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.skill_graph = nx.DiGraph()
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}

    async def add_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        category: str,
        prerequisites: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new skill with procedural knowledge."""
        # Create skill entry
        skill = {
            'id': skill_id,
            'name': name,
            'description': description,
            'category': category,
            'prerequisites': prerequisites or [],
            'proficiency': 0.0,
            'last_practiced': None,
            'practice_count': 0,
            'created_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Store skill
        self.skills[skill_id] = skill
        
        # Add to component memories
        await self.procedural_memory.add(skill_id, description, metadata)
        await self.semantic_memory.add(skill_id, description, metadata)
        
        # Add to skill graph
        self.skill_graph.add_node(skill_id, **skill)
        
        # Add prerequisites
        if prerequisites:
            for prereq_id in prerequisites:
                if prereq_id in self.skills:
                    self.skill_graph.add_edge(prereq_id, skill_id)
        
        # Initialize performance history
        self.performance_history[skill_id] = []

    async def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)

    async def get_skills_by_category(
        self,
        category: str,
        min_proficiency: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get skills in a specific category."""
        skills = []
        for skill_id, skill in self.skills.items():
            if skill['category'] == category:
                if min_proficiency is None or skill['proficiency'] >= min_proficiency:
                    skills.append(skill)
        return skills

    async def get_prerequisites(
        self,
        skill_id: str,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Get prerequisites for a skill."""
        if skill_id not in self.skill_graph:
            return []
            
        prerequisites = []
        for prereq_id in self.skill_graph.predecessors(skill_id):
            prereq = self.skills[prereq_id]
            if include_metadata:
                prerequisites.append(prereq)
            else:
                prerequisites.append({
                    'id': prereq_id,
                    'name': prereq['name'],
                    'proficiency': prereq['proficiency']
                })
        return prerequisites

    async def record_practice(
        self,
        skill_id: str,
        performance_score: float,
        duration: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a practice session for a skill."""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            
            # Update skill
            skill['last_practiced'] = datetime.now()
            skill['practice_count'] += 1
            
            # Calculate new proficiency
            old_proficiency = skill['proficiency']
            practice_impact = (performance_score - old_proficiency) * (1 - self.skill_decay)
            skill['proficiency'] = min(1.0, old_proficiency + practice_impact)
            
            # Record performance
            performance = {
                'timestamp': datetime.now(),
                'score': performance_score,
                'duration': duration,
                'context': context or {},
                'proficiency_before': old_proficiency,
                'proficiency_after': skill['proficiency']
            }
            self.performance_history[skill_id].append(performance)

    async def get_performance_history(
        self,
        skill_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get performance history for a skill."""
        if skill_id in self.performance_history:
            history = self.performance_history[skill_id]
            if limit:
                return history[-limit:]
            return history
        return []

    async def get_skill_progress(
        self,
        skill_id: str,
        time_window = None
    ) -> Dict[str, Any]:
        """Get progress statistics for a skill."""
        if skill_id not in self.skills:
            return {}
            
        skill = self.skills[skill_id]
        history = self.performance_history[skill_id]
        
        if time_window:
            cutoff = datetime.now() - time_window
            history = [h for h in history if h['timestamp'] >= cutoff]
        
        if not history:
            return {
                'current_proficiency': skill['proficiency'],
                'practice_count': skill['practice_count'],
                'last_practiced': skill['last_practiced']
            }
        
        return {
            'current_proficiency': skill['proficiency'],
            'practice_count': skill['practice_count'],
            'last_practiced': skill['last_practiced'],
            'avg_performance': np.mean([h['score'] for h in history]),
            'best_performance': max(h['score'] for h in history),
            'total_practice_time': sum(h['duration'] for h in history),
            'improvement_rate': (history[-1]['proficiency_after'] - history[0]['proficiency_before']) / len(history)
        }

    async def update_skill(
        self,
        skill_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update an existing skill."""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            skill.update(updates)
            
            # Update component memories
            if 'description' in updates:
                await self.procedural_memory.add(skill_id, updates['description'], skill['metadata'])
                await self.semantic_memory.add(skill_id, updates['description'], skill['metadata'])
            
            # Update graph
            self.skill_graph.nodes[skill_id].update(updates)

    async def remove_skill(self, skill_id: str) -> None:
        """Remove a skill."""
        if skill_id in self.skills:
            # Remove from component memories
            await self.procedural_memory.remove(skill_id)
            await self.semantic_memory.remove(skill_id)
            
            # Remove from graph
            self.skill_graph.remove_node(skill_id)
            
            # Remove performance history
            if skill_id in self.performance_history:
                del self.performance_history[skill_id]
            
            # Remove skill
            del self.skills[skill_id]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_skills': len(self.skills),
            'total_categories': len(set(s['category'] for s in self.skills.values())),
            'avg_proficiency': np.mean([s['proficiency'] for s in self.skills.values()]),
            'total_practice_sessions': sum(len(h) for h in self.performance_history.values()),
            'skill_graph_size': self.skill_graph.number_of_nodes(),
            'skill_graph_edges': self.skill_graph.number_of_edges()
        } 