"""
Multi-Agent Consensus Memory implementation using RAFT protocol.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import asyncio
from enum import Enum
from .base import BaseMemory
from .vector_store import VectorStoreMemory

class NodeState(Enum):
    """RAFT node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class LogEntry:
    """RAFT log entry."""
    def __init__(
        self,
        term: int,
        index: int,
        command: str,
        data: Dict[str, Any]
    ):
        self.term = term
        self.index = index
        self.command = command
        self.data = data
        self.timestamp = datetime.now()

class ConsensusMemory(BaseMemory):
    """Memory implementation using RAFT consensus protocol."""

    def __init__(
        self,
        node_id: str,
        nodes: List[str],
        election_timeout: float = 0.15,
        heartbeat_interval: float = 0.05,
        **kwargs
    ):
        """Initialize consensus memory."""
        super().__init__(**kwargs)
        
        # Node configuration
        self.node_id = node_id
        self.nodes = nodes
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # RAFT state
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index = defaultdict(lambda: 0)
        self.match_index = defaultdict(lambda: 0)
        
        # Component memories
        self.vector_memory = VectorStoreMemory()
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.consensus_state: Dict[str, Any] = defaultdict(dict)
        
        # Statistics
        self.total_entries = 0
        self.consensus_rounds = 0
        self.leader_changes = 0
        self.last_heartbeat = datetime.now()
        
        # Start background tasks
        asyncio.create_task(self._run_election_timer())
        asyncio.create_task(self._run_heartbeat())

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory through consensus."""
        if self.state == NodeState.LEADER:
            # Create log entry
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command="ADD_MEMORY",
                data={
                    'memory_id': memory_id,
                    'content': content,
                    'metadata': metadata
                }
            )
            
            # Append to log
            self.log.append(entry)
            
            # Replicate to followers
            await self._replicate_log()
            
            # Apply if committed
            if entry.index <= self.commit_index:
                await self._apply_entry(entry)
        else:
            # Forward to leader
            await self._forward_to_leader("ADD_MEMORY", {
                'memory_id': memory_id,
                'content': content,
                'metadata': metadata
            })

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory['access_count'] += 1
            memory['last_accessed'] = datetime.now()
            return memory
        return None

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update a memory through consensus."""
        if self.state == NodeState.LEADER:
            # Create log entry
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command="UPDATE_MEMORY",
                data={
                    'memory_id': memory_id,
                    'updates': updates
                }
            )
            
            # Append to log
            self.log.append(entry)
            
            # Replicate to followers
            await self._replicate_log()
            
            # Apply if committed
            if entry.index <= self.commit_index:
                await self._apply_entry(entry)
        else:
            # Forward to leader
            await self._forward_to_leader("UPDATE_MEMORY", {
                'memory_id': memory_id,
                'updates': updates
            })

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory through consensus."""
        if self.state == NodeState.LEADER:
            # Create log entry
            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                command="REMOVE_MEMORY",
                data={'memory_id': memory_id}
            )
            
            # Append to log
            self.log.append(entry)
            
            # Replicate to followers
            await self._replicate_log()
            
            # Apply if committed
            if entry.index <= self.commit_index:
                await self._apply_entry(entry)
        else:
            # Forward to leader
            await self._forward_to_leader("REMOVE_MEMORY", {
                'memory_id': memory_id
            })

    async def get_consensus_state(self) -> Dict[str, Any]:
        """Get current consensus state."""
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'log_length': len(self.log)
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': len(self.memories),
            'total_entries': self.total_entries,
            'consensus_rounds': self.consensus_rounds,
            'leader_changes': self.leader_changes,
            'current_state': self.state.value,
            'current_term': self.current_term,
            'commit_index': self.commit_index
        }

    async def _run_election_timer(self) -> None:
        """Run election timer for leader election."""
        while True:
            if self.state != NodeState.LEADER:
                # Check if election timeout
                if (datetime.now() - self.last_heartbeat).total_seconds() > self.election_timeout:
                    await self._start_election()
            await asyncio.sleep(self.election_timeout)

    async def _run_heartbeat(self) -> None:
        """Run heartbeat for leader."""
        while True:
            if self.state == NodeState.LEADER:
                await self._send_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)

    async def _start_election(self) -> None:
        """Start leader election."""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.leader_changes += 1
        
        # Request votes
        votes = 1  # Vote for self
        for node in self.nodes:
            if node != self.node_id:
                # This would typically send a request_vote RPC
                # For now, we'll simulate it
                if await self._request_vote(node):
                    votes += 1
        
        # Check if won election
        if votes > len(self.nodes) // 2:
            self.state = NodeState.LEADER
            self._initialize_leader_state()

    async def _request_vote(self, node: str) -> bool:
        """Request vote from a node."""
        # This would typically be an RPC call
        # For now, we'll simulate it
        return np.random.random() > 0.5

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to followers."""
        for node in self.nodes:
            if node != self.node_id:
                # This would typically send an append_entries RPC
                # For now, we'll simulate it
                await self._append_entries(node)
        self.last_heartbeat = datetime.now()

    async def _append_entries(self, node: str) -> bool:
        """Append entries to a follower."""
        # This would typically be an RPC call
        # For now, we'll simulate it
        return True

    async def _replicate_log(self) -> None:
        """Replicate log to followers."""
        for node in self.nodes:
            if node != self.node_id:
                await self._append_entries(node)
        self.consensus_rounds += 1

    async def _apply_entry(self, entry: LogEntry) -> None:
        """Apply a log entry."""
        if entry.command == "ADD_MEMORY":
            memory = {
                'id': entry.data['memory_id'],
                'content': entry.data['content'],
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 0,
                'metadata': entry.data['metadata']
            }
            self.memories[entry.data['memory_id']] = memory
            await self.vector_memory.add(
                entry.data['memory_id'],
                entry.data['content'],
                entry.data['metadata']
            )
            self.total_entries += 1
            
        elif entry.command == "UPDATE_MEMORY":
            if entry.data['memory_id'] in self.memories:
                memory = self.memories[entry.data['memory_id']]
                memory.update(entry.data['updates'])
                if 'content' in entry.data['updates']:
                    await self.vector_memory.add(
                        entry.data['memory_id'],
                        entry.data['updates']['content'],
                        memory['metadata']
                    )
                    
        elif entry.command == "REMOVE_MEMORY":
            if entry.data['memory_id'] in self.memories:
                del self.memories[entry.data['memory_id']]
                await self.vector_memory.remove(entry.data['memory_id'])
        
        self.last_applied = entry.index

    def _initialize_leader_state(self) -> None:
        """Initialize leader state."""
        for node in self.nodes:
            if node != self.node_id:
                self.next_index[node] = len(self.log)
                self.match_index[node] = 0

    async def _forward_to_leader(self, command: str, data: Dict[str, Any]) -> None:
        """Forward request to leader."""
        # This would typically forward to the current leader
        # For now, we'll just log it
        print(f"Forwarding {command} to leader: {data}") 