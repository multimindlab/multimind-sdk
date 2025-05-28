"""
Episodic memory implementation that stores and retrieves memories with temporal and spatial context.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from ..models.base import BaseLLM
from .base import BaseMemory

class EpisodicMemory(BaseMemory):
    """Memory that stores and retrieves episodic memories with temporal and spatial context."""

    def __init__(
        self,
        llm: BaseLLM,
        memory_key: str = "chat_history",
        storage_path: Optional[str] = None,
        max_episodes: int = 1000,
        temporal_decay_rate: float = 0.95,
        spatial_threshold: float = 0.7,
        emotional_weight: float = 0.3,
        temporal_weight: float = 0.3,
        spatial_weight: float = 0.4,
        min_confidence: float = 0.6,
        enable_consolidation: bool = True,
        consolidation_interval: int = 3600,  # 1 hour
        enable_chaining: bool = True,
        chain_depth: int = 3,
        importance_decay_rate: float = 0.98,
        emotional_analysis: bool = True,
        min_emotional_confidence: float = 0.7
    ):
        super().__init__(memory_key)
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_episodes = max_episodes
        self.temporal_decay_rate = temporal_decay_rate
        self.spatial_threshold = spatial_threshold
        self.emotional_weight = emotional_weight
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.min_confidence = min_confidence
        self.enable_consolidation = enable_consolidation
        self.consolidation_interval = consolidation_interval
        self.enable_chaining = enable_chaining
        self.chain_depth = chain_depth
        self.importance_decay_rate = importance_decay_rate
        self.emotional_analysis = emotional_analysis
        self.min_emotional_confidence = min_emotional_confidence
        
        # Initialize episode storage
        self.episodes: List[Dict[str, Any]] = []
        self.episode_embeddings: List[List[float]] = []
        self.spatial_index: Dict[str, Set[str]] = {}  # location -> episode_ids
        self.temporal_index: Dict[str, List[str]] = {}  # date -> episode_ids
        self.emotional_index: Dict[str, Set[str]] = {}  # emotion -> episode_ids
        self.episode_weights: Dict[str, float] = {}  # episode_id -> weight
        self.episode_chains: Dict[str, List[str]] = {}  # episode_id -> chain of related episode_ids
        self.episode_importance: Dict[str, float] = {}  # episode_id -> importance score
        self.emotional_profiles: Dict[str, Dict[str, float]] = {}  # episode_id -> emotion -> intensity
        self.last_consolidation = datetime.now()
        self.load()

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add message as a new episode with context."""
        # Create new episode
        episode_id = f"ep_{len(self.episodes)}"
        new_episode = {
            "id": episode_id,
            "content": message["content"],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "location": None,
                "emotions": set(),
                "participants": set(),
                "confidence": 1.0,
                "importance": 1.0,
                "consolidated": False,
                "emotional_intensity": 0.0,
                "chain_position": 0
            }
        }
        
        # Analyze episode
        await self._analyze_episode(new_episode)
        
        # Add to storage
        self.episodes.append(new_episode)
        self.episode_weights[episode_id] = 1.0
        self.episode_importance[episode_id] = 1.0
        
        # Update indices
        await self._update_indices(new_episode)
        
        # Update episode chains if enabled
        if self.enable_chaining:
            await self._update_episode_chains(new_episode)
        
        # Check for consolidation
        if self.enable_consolidation:
            current_time = datetime.now()
            if (current_time - self.last_consolidation).total_seconds() > self.consolidation_interval:
                await self._consolidate_episodes()
        
        # Maintain episode limit
        await self._maintain_episode_limit()
        
        await self.save()

    async def _analyze_episode(self, episode: Dict[str, Any]) -> None:
        """Analyze episode for metadata and context."""
        try:
            # Analyze episode content
            prompt = f"""
            Analyze the following episode and determine:
            1. Location or setting
            2. Emotions expressed
            3. Participants involved
            4. Confidence in analysis (0-1)
            5. Importance of the episode (0-1)
            6. Emotional intensity (0-1)
            
            Episode: {episode['content']}
            
            Return in format:
            Location: <location>
            Emotions: <comma-separated emotions>
            Participants: <comma-separated participants>
            Confidence: <confidence score>
            Importance: <importance score>
            Emotional Intensity: <intensity score>
            """
            response = await self.llm.generate(prompt)
            
            # Parse response
            lines = response.split('\n')
            for line in lines:
                if line.startswith('Location:'):
                    episode['metadata']['location'] = line.split(':', 1)[1].strip()
                elif line.startswith('Emotions:'):
                    emotions = line.split(':', 1)[1].strip().split(',')
                    episode['metadata']['emotions'] = {e.strip() for e in emotions}
                elif line.startswith('Participants:'):
                    participants = line.split(':', 1)[1].strip().split(',')
                    episode['metadata']['participants'] = {p.strip() for p in participants}
                elif line.startswith('Confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())
                    episode['metadata']['confidence'] = confidence
                elif line.startswith('Importance:'):
                    importance = float(line.split(':', 1)[1].strip())
                    episode['metadata']['importance'] = importance
                elif line.startswith('Emotional Intensity:'):
                    intensity = float(line.split(':', 1)[1].strip())
                    episode['metadata']['emotional_intensity'] = intensity
            
            # Get episode embedding
            embedding = await self.llm.embeddings(episode['content'])
            self.episode_embeddings.append(embedding)
            
            # Analyze emotional profile if enabled
            if self.emotional_analysis:
                await self._analyze_emotional_profile(episode)
            
        except Exception as e:
            print(f"Error analyzing episode: {e}")

    async def _analyze_emotional_profile(self, episode: Dict[str, Any]) -> None:
        """Analyze emotional profile of an episode."""
        try:
            prompt = f"""
            Analyze the emotional profile of this episode and determine the intensity (0-1) of each emotion:
            
            Episode: {episode['content']}
            
            Return in format:
            Emotion: <emotion name>
            Intensity: <intensity score>
            ---
            """
            response = await self.llm.generate(prompt)
            
            emotional_profile = {}
            current_emotion = None
            
            for line in response.split('\n'):
                if line.startswith('Emotion:'):
                    current_emotion = line.split(':', 1)[1].strip()
                elif line.startswith('Intensity:'):
                    intensity = float(line.split(':', 1)[1].strip())
                    if current_emotion:
                        emotional_profile[current_emotion] = intensity
            
            self.emotional_profiles[episode['id']] = emotional_profile
            
        except Exception as e:
            print(f"Error analyzing emotional profile: {e}")

    async def _update_episode_chains(self, episode: Dict[str, Any]) -> None:
        """Update episode chains with the new episode."""
        if not self.episodes:
            self.episode_chains[episode['id']] = []
            return
        
        # Find most related episode
        related_episode = await self._find_most_related_episode(episode)
        
        if related_episode:
            # Add to existing chain
            chain = self.episode_chains.get(related_episode['id'], [])
            if len(chain) < self.chain_depth:
                chain.append(episode['id'])
                self.episode_chains[episode['id']] = chain
                episode['metadata']['chain_position'] = len(chain)
        else:
            # Start new chain
            self.episode_chains[episode['id']] = []

    async def _find_most_related_episode(
        self,
        episode: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the most related episode to the given episode."""
        if not self.episodes:
            return None
        
        # Get episode embedding
        episode_embedding = await self.llm.embeddings(episode['content'])
        
        # Calculate similarities
        similarities = []
        for i, existing_embedding in enumerate(self.episode_embeddings):
            similarity = self._cosine_similarity(episode_embedding, existing_embedding)
            if similarity >= self.spatial_threshold:
                similarities.append({
                    "episode": self.episodes[i],
                    "similarity": similarity
                })
        
        if not similarities:
            return None
        
        return max(similarities, key=lambda x: x["similarity"])["episode"]

    async def get_episode_chain(
        self,
        episode_id: str,
        max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get chain of related episodes."""
        if episode_id not in self.episode_chains:
            return []
        
        chain = []
        for related_id in self.episode_chains[episode_id]:
            if max_depth is None or len(chain) < max_depth:
                episode = await self.get_episode_by_id(related_id)
                if episode:
                    chain.append(episode)
        
        return chain

    async def get_episode_by_id(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get an episode by its ID."""
        try:
            return next(ep for ep in self.episodes if ep['id'] == episode_id)
        except StopIteration:
            return None

    async def get_emotional_profile(
        self,
        episode_id: str,
        min_intensity: Optional[float] = None
    ) -> Dict[str, float]:
        """Get emotional profile of an episode."""
        if episode_id not in self.emotional_profiles:
            return {}
        
        if min_intensity is None:
            return self.emotional_profiles[episode_id]
        
        return {
            emotion: intensity
            for emotion, intensity in self.emotional_profiles[episode_id].items()
            if intensity >= min_intensity
        }

    async def _update_indices(self, episode: Dict[str, Any]) -> None:
        """Update spatial, temporal, and emotional indices."""
        # Update spatial index
        location = episode['metadata']['location']
        if location:
            if location not in self.spatial_index:
                self.spatial_index[location] = set()
            self.spatial_index[location].add(episode['id'])
        
        # Update temporal index
        date = episode['timestamp'].split('T')[0]
        if date not in self.temporal_index:
            self.temporal_index[date] = []
        self.temporal_index[date].append(episode['id'])
        
        # Update emotional index
        for emotion in episode['metadata']['emotions']:
            if emotion not in self.emotional_index:
                self.emotional_index[emotion] = set()
            self.emotional_index[emotion].add(episode['id'])

    async def _consolidate_episodes(self) -> None:
        """Consolidate similar episodes to reduce redundancy."""
        # Find similar episodes
        for i, episode1 in enumerate(self.episodes):
            if episode1['metadata']['consolidated']:
                continue
            
            for j, episode2 in enumerate(self.episodes[i+1:], i+1):
                if episode2['metadata']['consolidated']:
                    continue
                
                # Check similarity
                similarity = self._cosine_similarity(
                    self.episode_embeddings[i],
                    self.episode_embeddings[j]
                )
                
                if similarity >= self.spatial_threshold:
                    # Consolidate episodes
                    await self._merge_episodes(episode1['id'], episode2['id'])
        
        self.last_consolidation = datetime.now()

    async def _merge_episodes(self, episode_id1: str, episode_id2: str) -> None:
        """Merge two similar episodes."""
        episode1 = next(ep for ep in self.episodes if ep['id'] == episode_id1)
        episode2 = next(ep for ep in self.episodes if ep['id'] == episode_id2)
        
        # Merge content
        merged_content = f"{episode1['content']}\n{episode2['content']}"
        
        # Update episode1
        episode1['content'] = merged_content
        episode1['metadata']['emotions'].update(episode2['metadata']['emotions'])
        episode1['metadata']['participants'].update(episode2['metadata']['participants'])
        episode1['metadata']['confidence'] = min(
            episode1['metadata']['confidence'],
            episode2['metadata']['confidence']
        )
        episode1['metadata']['importance'] = max(
            episode1['metadata']['importance'],
            episode2['metadata']['importance']
        )
        episode1['metadata']['consolidated'] = True
        
        # Update embedding
        idx1 = next(i for i, ep in enumerate(self.episodes) if ep['id'] == episode_id1)
        self.episode_embeddings[idx1] = await self.llm.embeddings(merged_content)
        
        # Remove episode2
        await self._remove_episode(episode_id2)

    async def _maintain_episode_limit(self) -> None:
        """Maintain episode limit by removing least important episodes."""
        if len(self.episodes) > self.max_episodes:
            # Sort episodes by weight
            sorted_episodes = sorted(
                self.episodes,
                key=lambda x: self.episode_weights[x['id']]
            )
            
            # Remove episodes with lowest weights
            episodes_to_remove = sorted_episodes[:len(self.episodes) - self.max_episodes]
            for episode in episodes_to_remove:
                await self._remove_episode(episode['id'])

    async def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode and update indices."""
        # Remove from episodes
        episode_idx = next(i for i, ep in enumerate(self.episodes) if ep['id'] == episode_id)
        self.episodes.pop(episode_idx)
        self.episode_embeddings.pop(episode_idx)
        
        # Remove from indices
        for location in self.spatial_index:
            self.spatial_index[location].discard(episode_id)
        
        for date in self.temporal_index:
            if episode_id in self.temporal_index[date]:
                self.temporal_index[date].remove(episode_id)
        
        for emotion in self.emotional_index:
            self.emotional_index[emotion].discard(episode_id)
        
        # Remove weight
        del self.episode_weights[episode_id]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages from all episodes."""
        messages = []
        for episode in self.episodes:
            messages.append({
                "role": "episode",
                "content": episode['content'],
                "timestamp": episode['timestamp']
            })
        return sorted(messages, key=lambda x: x["timestamp"])

    async def clear(self) -> None:
        """Clear all episodes."""
        self.episodes = []
        self.episode_embeddings = []
        self.spatial_index = {}
        self.temporal_index = {}
        self.emotional_index = {}
        self.episode_weights = {}
        self.episode_chains = {}
        self.episode_importance = {}
        self.emotional_profiles = {}
        await self.save()

    async def save(self) -> None:
        """Save episodes to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({
                    "episodes": self.episodes,
                    "spatial_index": {
                        k: list(v) for k, v in self.spatial_index.items()
                    },
                    "temporal_index": self.temporal_index,
                    "emotional_index": {
                        k: list(v) for k, v in self.emotional_index.items()
                    },
                    "episode_weights": self.episode_weights,
                    "episode_chains": self.episode_chains,
                    "episode_importance": self.episode_importance,
                    "emotional_profiles": self.emotional_profiles,
                    "last_consolidation": self.last_consolidation.isoformat()
                }, f)

    def load(self) -> None:
        """Load episodes from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.episodes = data.get("episodes", [])
                self.spatial_index = {
                    k: set(v) for k, v in data.get("spatial_index", {}).items()
                }
                self.temporal_index = data.get("temporal_index", {})
                self.emotional_index = {
                    k: set(v) for k, v in data.get("emotional_index", {}).items()
                }
                self.episode_weights = data.get("episode_weights", {})
                self.episode_chains = data.get("episode_chains", {})
                self.episode_importance = data.get("episode_importance", {})
                self.emotional_profiles = data.get("emotional_profiles", {})
                self.last_consolidation = datetime.fromisoformat(
                    data.get("last_consolidation", datetime.now().isoformat())
                )
                
                # Recreate embeddings
                self.episode_embeddings = []
                for episode in self.episodes:
                    self.episode_embeddings.append(
                        self.llm.embeddings(episode['content'])
                    )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    async def get_episodes_by_location(
        self,
        location: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get episodes from a specific location."""
        if location not in self.spatial_index:
            return []
        
        episodes = []
        for episode_id in self.spatial_index[location]:
            episode = next(ep for ep in self.episodes if ep['id'] == episode_id)
            if min_confidence is None or episode['metadata']['confidence'] >= min_confidence:
                episodes.append(episode)
        
        return sorted(episodes, key=lambda x: x['timestamp'])

    async def get_episodes_by_emotion(
        self,
        emotion: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get episodes with a specific emotion."""
        if emotion not in self.emotional_index:
            return []
        
        episodes = []
        for episode_id in self.emotional_index[emotion]:
            episode = next(ep for ep in self.episodes if ep['id'] == episode_id)
            if min_confidence is None or episode['metadata']['confidence'] >= min_confidence:
                episodes.append(episode)
        
        return sorted(episodes, key=lambda x: x['timestamp'])

    async def get_episodes_by_date(
        self,
        date: str,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get episodes from a specific date."""
        if date not in self.temporal_index:
            return []
        
        episodes = []
        for episode_id in self.temporal_index[date]:
            episode = next(ep for ep in self.episodes if ep['id'] == episode_id)
            if min_confidence is None or episode['metadata']['confidence'] >= min_confidence:
                episodes.append(episode)
        
        return sorted(episodes, key=lambda x: x['timestamp'])

    async def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics about episodes."""
        stats = {
            "total_episodes": len(self.episodes),
            "location_distribution": {},
            "emotion_distribution": {},
            "participant_distribution": {},
            "confidence_distribution": {
                "high": 0,  # > 0.8
                "medium": 0,  # 0.5-0.8
                "low": 0  # < 0.5
            },
            "importance_distribution": {
                "high": 0,  # > 0.7
                "medium": 0,  # 0.3-0.7
                "low": 0  # < 0.3
            },
            "consolidation_stats": {
                "consolidated": 0,
                "unconsolidated": 0
            },
            "chain_stats": {
                "total_chains": len(self.episode_chains),
                "max_chain_length": max(
                    (len(chain) for chain in self.episode_chains.values()),
                    default=0
                ),
                "average_chain_length": sum(
                    len(chain) for chain in self.episode_chains.values()
                ) / len(self.episode_chains) if self.episode_chains else 0
            },
            "emotional_stats": {
                "total_emotional_profiles": len(self.emotional_profiles),
                "average_intensity": sum(
                    sum(profile.values()) / len(profile)
                    for profile in self.emotional_profiles.values()
                ) / len(self.emotional_profiles) if self.emotional_profiles else 0
            }
        }
        
        for episode in self.episodes:
            # Count locations
            location = episode['metadata']['location']
            if location:
                stats["location_distribution"][location] = \
                    stats["location_distribution"].get(location, 0) + 1
            
            # Count emotions
            for emotion in episode['metadata']['emotions']:
                stats["emotion_distribution"][emotion] = \
                    stats["emotion_distribution"].get(emotion, 0) + 1
            
            # Count participants
            for participant in episode['metadata']['participants']:
                stats["participant_distribution"][participant] = \
                    stats["participant_distribution"].get(participant, 0) + 1
            
            # Count confidence levels
            confidence = episode['metadata']['confidence']
            if confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif confidence > 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
            
            # Count importance levels
            importance = episode['metadata']['importance']
            if importance > 0.7:
                stats["importance_distribution"]["high"] += 1
            elif importance > 0.3:
                stats["importance_distribution"]["medium"] += 1
            else:
                stats["importance_distribution"]["low"] += 1
            
            # Count consolidation status
            if episode['metadata']['consolidated']:
                stats["consolidation_stats"]["consolidated"] += 1
            else:
                stats["consolidation_stats"]["unconsolidated"] += 1
        
        return stats

    async def get_episode_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for episode optimization."""
        suggestions = []
        
        # Check episode count
        if len(self.episodes) > self.max_episodes * 0.8:
            suggestions.append({
                "type": "episode_limit",
                "suggestion": "Consider increasing max_episodes or consolidating similar episodes"
            })
        
        # Check confidence distribution
        stats = await self.get_episode_stats()
        if stats["confidence_distribution"]["low"] > len(self.episodes) * 0.3:
            suggestions.append({
                "type": "confidence_quality",
                "suggestion": "Consider improving episode analysis quality"
            })
        
        # Check consolidation status
        if stats["consolidation_stats"]["unconsolidated"] > len(self.episodes) * 0.5:
            suggestions.append({
                "type": "consolidation",
                "suggestion": "Consider running episode consolidation"
            })
        
        # Check location diversity
        if len(stats["location_distribution"]) < 3:
            suggestions.append({
                "type": "location_diversity",
                "suggestion": "Consider adding more diverse locations"
            })
        
        # Check chain statistics
        if stats["chain_stats"]["average_chain_length"] < 2:
            suggestions.append({
                "type": "chain_development",
                "suggestion": "Consider developing longer episode chains"
            })
        
        # Check emotional analysis
        if len(stats["emotional_stats"]["total_emotional_profiles"]) < len(self.episodes) * 0.5:
            suggestions.append({
                "type": "emotional_analysis",
                "suggestion": "Consider enabling emotional analysis for more episodes"
            })
        
        return suggestions 