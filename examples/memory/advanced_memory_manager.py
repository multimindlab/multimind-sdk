"""
Advanced Memory Manager with comprehensive features for enterprise use.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from multimind.memory import (
    HybridMemory,
    ConversationBufferMemory,
    VectorStoreMemory,
    FastWeightMemory
)

class MemoryPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AccessLevel(Enum):
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    RESTRICTED = 4

@dataclass
class RetentionPolicy:
    max_age: timedelta
    max_size: int
    priority: MemoryPriority
    access_level: AccessLevel
    auto_archive: bool = False
    require_encryption: bool = False

@dataclass
class UserProfile:
    user_id: str
    access_level: AccessLevel
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]

class AdvancedMemoryManager:
    def __init__(
        self,
        retention_policies: Dict[str, RetentionPolicy],
        debug_mode: bool = False,
        trace_enabled: bool = False
    ):
        self.memory_system = HybridMemory(
            memories=[
                ConversationBufferMemory(),
                VectorStoreMemory(),
                FastWeightMemory(
                    input_size=768,
                    memory_size=1024
                )
            ]
        )
        self.retention_policies = retention_policies
        self.user_profiles: Dict[str, UserProfile] = {}
        self.debug_mode = debug_mode
        self.trace_enabled = trace_enabled
        self.event_hooks: Dict[str, List[Callable]] = {}
        self.audit_log: List[Dict] = []
        
    async def add_memory(
        self,
        memory_id: str,
        content: Any,
        metadata: Dict[str, Any],
        user_id: str,
        policy_name: str
    ) -> None:
        """Add memory with policy enforcement and access control."""
        # Check user access
        if not self._check_user_access(user_id, policy_name):
            raise PermissionError(f"User {user_id} not authorized for this operation")
        
        # Apply retention policy
        policy = self.retention_policies[policy_name]
        metadata.update({
            "created_at": datetime.now(),
            "expires_at": datetime.now() + policy.max_age,
            "priority": policy.priority,
            "access_level": policy.access_level
        })
        
        # Add memory
        await self.memory_system.add_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata
        )
        
        # Log operation
        self._log_operation("add_memory", user_id, memory_id, metadata)
        
        # Trigger hooks
        await self._trigger_hooks("memory_added", {
            "memory_id": memory_id,
            "user_id": user_id,
            "metadata": metadata
        })
    
    async def get_memory(
        self,
        memory_id: str,
        user_id: str,
        query: Optional[str] = None
    ) -> Dict:
        """Retrieve memory with access control and audit logging."""
        # Check access
        memory = await self.memory_system.get_memory(memory_id, query)
        if not self._check_memory_access(user_id, memory["metadata"]):
            raise PermissionError(f"User {user_id} not authorized to access this memory")
        
        # Log access
        self._log_operation("get_memory", user_id, memory_id, memory["metadata"])
        
        return memory
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> None:
        """Update memory with conflict resolution and versioning."""
        # Get current memory
        current = await self.memory_system.get_memory(memory_id)
        
        # Check access
        if not self._check_memory_access(user_id, current["metadata"]):
            raise PermissionError(f"User {user_id} not authorized to update this memory")
        
        # Resolve conflicts
        resolved_updates = self._resolve_conflicts(current, updates)
        
        # Update memory
        await self.memory_system.update_memory(
            memory_id=memory_id,
            updates=resolved_updates
        )
        
        # Log update
        self._log_operation("update_memory", user_id, memory_id, resolved_updates)
        
        # Trigger hooks
        await self._trigger_hooks("memory_updated", {
            "memory_id": memory_id,
            "user_id": user_id,
            "updates": resolved_updates
        })
    
    async def enforce_retention_policies(self) -> None:
        """Enforce retention policies and clean up expired memories."""
        for memory_id, memory in await self.memory_system.get_all_memories():
            policy = self.retention_policies.get(memory["metadata"].get("policy_name"))
            if policy:
                if self._should_evict(memory, policy):
                    await self._evict_memory(memory_id, memory["metadata"])
    
    def register_user(self, user_profile: UserProfile) -> None:
        """Register a new user profile."""
        self.user_profiles[user_profile.user_id] = user_profile
    
    def add_event_hook(self, event: str, hook: Callable) -> None:
        """Register an event hook."""
        if event not in self.event_hooks:
            self.event_hooks[event] = []
        self.event_hooks[event].append(hook)
    
    def _check_user_access(self, user_id: str, policy_name: str) -> bool:
        """Check if user has access based on policy."""
        user = self.user_profiles.get(user_id)
        policy = self.retention_policies.get(policy_name)
        return user and policy and user.access_level.value >= policy.access_level.value
    
    def _check_memory_access(self, user_id: str, metadata: Dict) -> bool:
        """Check if user has access to specific memory."""
        user = self.user_profiles.get(user_id)
        return user and user.access_level.value >= metadata["access_level"].value
    
    def _resolve_conflicts(self, current: Dict, updates: Dict) -> Dict:
        """Resolve conflicts between current and updated content."""
        resolved = current.copy()
        for key, value in updates.items():
            if key in current and current[key] != value:
                # Implement conflict resolution strategy
                if isinstance(value, (int, float)):
                    resolved[key] = max(current[key], value)
                elif isinstance(value, str):
                    resolved[key] = f"{current[key]} | {value}"
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved
    
    def _should_evict(self, memory: Dict, policy: RetentionPolicy) -> bool:
        """Determine if memory should be evicted based on policy."""
        created_at = memory["metadata"]["created_at"]
        return (
            datetime.now() > created_at + policy.max_age or
            memory["metadata"]["priority"].value < policy.priority.value
        )
    
    async def _evict_memory(self, memory_id: str, metadata: Dict) -> None:
        """Evict memory based on policy."""
        if metadata.get("auto_archive"):
            # Archive memory instead of deletion
            await self.memory_system.archive_memory(memory_id)
        else:
            await self.memory_system.delete_memory(memory_id)
        
        # Log eviction
        self._log_operation("evict_memory", "system", memory_id, metadata)
    
    def _log_operation(self, operation: str, user_id: str, memory_id: str, metadata: Dict) -> None:
        """Log memory operation for audit trail."""
        log_entry = {
            "timestamp": datetime.now(),
            "operation": operation,
            "user_id": user_id,
            "memory_id": memory_id,
            "metadata": metadata
        }
        self.audit_log.append(log_entry)
        
        if self.debug_mode:
            print(f"DEBUG: {log_entry}")
    
    async def _trigger_hooks(self, event: str, data: Dict) -> None:
        """Trigger registered event hooks."""
        for hook in self.event_hooks.get(event, []):
            try:
                await hook(data)
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in hook {hook.__name__}: {e}")

async def example_usage():
    """Demonstrate advanced memory manager features."""
    # Define retention policies
    policies = {
        "standard": RetentionPolicy(
            max_age=timedelta(days=30),
            max_size=1000,
            priority=MemoryPriority.MEDIUM,
            access_level=AccessLevel.INTERNAL
        ),
        "critical": RetentionPolicy(
            max_age=timedelta(days=90),
            max_size=5000,
            priority=MemoryPriority.CRITICAL,
            access_level=AccessLevel.RESTRICTED,
            require_encryption=True
        )
    }
    
    # Create memory manager
    manager = AdvancedMemoryManager(
        retention_policies=policies,
        debug_mode=True,
        trace_enabled=True
    )
    
    # Register users
    manager.register_user(UserProfile(
        user_id="admin",
        access_level=AccessLevel.RESTRICTED,
        preferences={"language": "en"},
        metadata={"role": "administrator"}
    ))
    
    manager.register_user(UserProfile(
        user_id="user1",
        access_level=AccessLevel.INTERNAL,
        preferences={"language": "en"},
        metadata={"role": "developer"}
    ))
    
    # Add event hooks
    async def on_memory_added(data: Dict):
        print(f"Memory added: {data['memory_id']}")
    
    manager.add_event_hook("memory_added", on_memory_added)
    
    # Add memories
    await manager.add_memory(
        memory_id="critical_doc_1",
        content="Sensitive information",
        metadata={"type": "document"},
        user_id="admin",
        policy_name="critical"
    )
    
    await manager.add_memory(
        memory_id="standard_doc_1",
        content="Regular information",
        metadata={"type": "document"},
        user_id="user1",
        policy_name="standard"
    )
    
    # Retrieve memory
    memory = await manager.get_memory(
        memory_id="critical_doc_1",
        user_id="admin"
    )
    print("Retrieved memory:", memory)
    
    # Update memory
    await manager.update_memory(
        memory_id="standard_doc_1",
        updates={"content": "Updated information"},
        user_id="user1"
    )
    
    # Enforce retention policies
    await manager.enforce_retention_policies()
    
    # Print audit log
    print("\nAudit Log:")
    for entry in manager.audit_log:
        print(entry)

if __name__ == "__main__":
    asyncio.run(example_usage()) 