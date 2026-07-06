"""
Consensus Memory: local in-process replication with majority-vote reads.

Networked RAFT consensus (leader election, vote/append RPCs, forwarding to a
remote leader) is NOT implemented; those methods raise NotImplementedError
instead of simulating agreement.
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseMemory

logger = logging.getLogger(__name__)

_NETWORK_RAFT_ERROR = (
    "Networked RAFT consensus (leader election, RPCs) is not implemented. "
    "ConsensusMemory supports local in-process replication only: call "
    "attach_replicas([...]) to form a local cluster."
)


class NodeState(Enum):
    """Consensus node states."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntry:
    """Replicated log entry."""

    def __init__(self, term: int, index: int, command: str, data: Dict[str, Any]):
        self.term = term
        self.index = index
        self.command = command
        self.data = data
        self.timestamp = datetime.now()


class ConsensusMemory(BaseMemory):
    """
    Memory replicated across in-process nodes with majority-vote reads.

    A node standing alone (or after attach_replicas) acts as leader of its
    local cluster and synchronously replicates every write to the attached
    replicas. Reads can be checked across replicas with get_majority_memory,
    which returns only values a strict majority of nodes agree on.
    """

    def __init__(
        self,
        node_id: str,
        nodes: Optional[List[str]] = None,
        storage_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize consensus memory."""
        super().__init__(**kwargs)

        self.node_id = node_id
        self.nodes = nodes or [node_id]
        self.storage_path = Path(storage_path) if storage_path else None

        # Consensus state
        self.state = NodeState.LEADER if len(self.nodes) <= 1 else NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = -1
        self.last_applied = -1

        # Local in-process replicas
        self.replicas: List["ConsensusMemory"] = []

        # Leader bookkeeping
        self.next_index = defaultdict(lambda: 0)
        self.match_index = defaultdict(lambda: 0)

        # Memory storage
        self.memories: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_entries = 0
        self.consensus_rounds = 0
        self.leader_changes = 0

    # --- Local in-process cluster ---

    def attach_replicas(self, replicas: List["ConsensusMemory"]) -> None:
        """Form a local in-process cluster with this node as leader."""
        self.replicas = [r for r in replicas if r is not self]
        if self.state != NodeState.LEADER:
            self.state = NodeState.LEADER
            self.leader_changes += 1
        self.nodes = [self.node_id] + [r.node_id for r in self.replicas]
        for replica in self.replicas:
            replica.state = NodeState.FOLLOWER

    async def add_memory(
        self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory, replicated to all attached in-process replicas."""
        await self._commit(
            "ADD_MEMORY", {"memory_id": memory_id, "content": content, "metadata": metadata}
        )

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """Update a memory, replicated to all attached in-process replicas."""
        await self._commit("UPDATE_MEMORY", {"memory_id": memory_id, "updates": updates})

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory, replicated to all attached in-process replicas."""
        await self._commit("REMOVE_MEMORY", {"memory_id": memory_id})

    async def _commit(self, command: str, data: Dict[str, Any]) -> None:
        """Append, replicate, and apply a log entry on the local cluster."""
        if self.state != NodeState.LEADER:
            raise NotImplementedError(_NETWORK_RAFT_ERROR)

        entry = LogEntry(term=self.current_term, index=len(self.log), command=command, data=data)
        self.log.append(entry)
        acks = 1
        for replica in self.replicas:
            replica.log.append(entry)
            await replica._apply_entry(entry)
            replica.commit_index = entry.index
            acks += 1
        # In-process replication is synchronous, so majority always holds.
        if acks > len(self.nodes) // 2:
            self.commit_index = entry.index
            await self._apply_entry(entry)
        self.consensus_rounds += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID from this node's local state."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now()
            return memory
        return None

    async def get_majority_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Majority-vote read: return the content a strict majority of the local
        cluster (this node + attached replicas) agrees on, or None.
        """
        nodes = [self] + self.replicas
        votes = Counter()
        for node in nodes:
            memory = node.memories.get(memory_id)
            if memory is not None:
                votes[memory["content"]] += 1
        if not votes:
            return None
        content, count = votes.most_common(1)[0]
        if count > len(nodes) // 2:
            return {"memory_id": memory_id, "content": content, "votes": count, "nodes": len(nodes)}
        return None

    async def get_consensus_state(self) -> Dict[str, Any]:
        """Get current consensus state."""
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "voted_for": self.voted_for,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_length": len(self.log),
            "replicas": [r.node_id for r in self.replicas],
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": len(self.memories),
            "total_entries": self.total_entries,
            "consensus_rounds": self.consensus_rounds,
            "leader_changes": self.leader_changes,
            "current_state": self.state.value,
            "current_term": self.current_term,
            "commit_index": self.commit_index,
        }

    async def _apply_entry(self, entry: LogEntry) -> None:
        """Apply a log entry to local state."""
        if entry.command == "ADD_MEMORY":
            self.memories[entry.data["memory_id"]] = {
                "id": entry.data["memory_id"],
                "content": entry.data["content"],
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "access_count": 0,
                "metadata": entry.data["metadata"],
            }
            self.total_entries += 1

        elif entry.command == "UPDATE_MEMORY":
            if entry.data["memory_id"] in self.memories:
                self.memories[entry.data["memory_id"]].update(entry.data["updates"])

        elif entry.command == "REMOVE_MEMORY":
            self.memories.pop(entry.data["memory_id"], None)

        self.last_applied = entry.index

    # --- BaseMemory interface ---

    async def add_message(self, message: Dict[str, str]) -> None:
        """Add a message as a replicated memory."""
        memory_id = f"message_{len(self.log)}"
        await self.add_memory(
            memory_id, message["content"], {"role": message.get("role", "user")}
        )

    async def get_messages(self) -> List[Dict[str, str]]:
        """Get all stored memories as messages."""
        messages = []
        for memory in self.memories.values():
            metadata = memory.get("metadata") or {}
            messages.append(
                {
                    "role": metadata.get("role", "consensus_memory"),
                    "content": memory["content"],
                }
            )
        return messages

    async def clear(self) -> None:
        """Clear local state (does not touch replicas)."""
        self.memories = {}
        self.log = []
        self.commit_index = -1
        self.last_applied = -1
        await self.save()

    async def save(self) -> None:
        """Save memories to persistent storage."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                memory_id: {
                    "id": memory["id"],
                    "content": memory["content"],
                    "metadata": memory["metadata"],
                    "access_count": memory["access_count"],
                }
                for memory_id, memory in self.memories.items()
            }
            with open(self.storage_path, "w") as f:
                json.dump({"node_id": self.node_id, "memories": serializable}, f)

    async def load(self) -> None:
        """Load memories from persistent storage."""
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for memory_id, memory in data.get("memories", {}).items():
                self.memories[memory_id] = {
                    "id": memory["id"],
                    "content": memory["content"],
                    "created_at": datetime.now(),
                    "last_accessed": datetime.now(),
                    "access_count": memory.get("access_count", 0),
                    "metadata": memory.get("metadata"),
                }

    # --- Networked RAFT: not implemented, fail honest ---

    async def start_background_tasks(self) -> None:
        """Networked RAFT election/heartbeat loops are not implemented."""
        raise NotImplementedError(_NETWORK_RAFT_ERROR)

    async def stop_background_tasks(self) -> None:
        """No background tasks run; nothing to stop."""
        return None

    async def _start_election(self) -> None:
        raise NotImplementedError(_NETWORK_RAFT_ERROR)

    async def _request_vote(self, node: str) -> bool:
        raise NotImplementedError(_NETWORK_RAFT_ERROR)

    async def _send_heartbeat(self) -> None:
        raise NotImplementedError(_NETWORK_RAFT_ERROR)

    async def _append_entries(self, node: str) -> bool:
        raise NotImplementedError(_NETWORK_RAFT_ERROR)

    async def _forward_to_leader(self, command: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError(_NETWORK_RAFT_ERROR)
