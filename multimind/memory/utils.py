"""
Utility functions for memory management.
"""

from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
import json
from pathlib import Path
import pickle
from .base import BaseMemory

class MemoryUtils:
    """Utility functions for memory management."""

    @staticmethod
    async def save_memory(
        memory: BaseMemory,
        path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save memory to disk.
        
        Args:
            memory: Memory instance to save
            path: Path to save to
            format: Save format (json or pickle)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get memory state
        state = {
            "messages": memory.messages,
            "metadata": memory.metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save based on format
        if format == "json":
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        else:  # pickle
            with open(path, "wb") as f:
                pickle.dump(state, f)

    @staticmethod
    async def load_memory(
        memory_class: Type[BaseMemory],
        path: Union[str, Path],
        format: str = "json",
        **kwargs
    ) -> BaseMemory:
        """
        Load memory from disk.
        
        Args:
            memory_class: Memory class to instantiate
            path: Path to load from
            format: Load format (json or pickle)
            **kwargs: Additional arguments for memory class
            
        Returns:
            Loaded memory instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")
            
        # Load based on format
        if format == "json":
            with open(path, "r") as f:
                state = json.load(f)
        else:  # pickle
            with open(path, "rb") as f:
                state = pickle.load(f)
                
        # Create memory instance
        memory = memory_class(**kwargs)
        
        # Restore state
        memory.messages = state["messages"]
        memory.metadata = state["metadata"]
        
        return memory

    @staticmethod
    async def merge_memories(
        memories: List[BaseMemory],
        strategy: str = "append"  # append, interleave, smart
    ) -> BaseMemory:
        """
        Merge multiple memories into one.
        
        Args:
            memories: List of memories to merge
            strategy: Merge strategy
            
        Returns:
            Merged memory instance
        """
        if not memories:
            raise ValueError("No memories to merge")
            
        # Create new memory of same type as first memory
        merged = type(memories[0])()
        
        if strategy == "append":
            # Simply append all messages
            for memory in memories:
                for msg in memory.messages:
                    await merged.add_message(
                        msg["message"],
                        msg["metadata"]
                    )
                    
        elif strategy == "interleave":
            # Interleave messages by timestamp
            all_messages = []
            for memory in memories:
                all_messages.extend(memory.messages)
                
            # Sort by timestamp
            all_messages.sort(
                key=lambda x: x["timestamp"]
            )
            
            # Add to merged memory
            for msg in all_messages:
                await merged.add_message(
                    msg["message"],
                    msg["metadata"]
                )
                
        else:  # smart
            # Smart merge based on content similarity
            # This is a simplified implementation
            # In practice, you would use more sophisticated merging
            seen_content = set()
            
            for memory in memories:
                for msg in memory.messages:
                    content = msg["message"].get("content", "")
                    if content not in seen_content:
                        await merged.add_message(
                            msg["message"],
                            msg["metadata"]
                        )
                        seen_content.add(content)
                        
        return merged

    @staticmethod
    async def filter_memory(
        memory: BaseMemory,
        filter_func: callable,
        **kwargs
    ) -> BaseMemory:
        """
        Filter memory based on a function.
        
        Args:
            memory: Memory to filter
            filter_func: Function to filter messages
            **kwargs: Additional arguments for filter function
            
        Returns:
            Filtered memory instance
        """
        # Create new memory of same type
        filtered = type(memory)()
        
        # Filter messages
        for msg in memory.messages:
            if filter_func(msg, **kwargs):
                await filtered.add_message(
                    msg["message"],
                    msg["metadata"]
                )
                
        return filtered

    @staticmethod
    async def transform_memory(
        memory: BaseMemory,
        transform_func: callable,
        **kwargs
    ) -> BaseMemory:
        """
        Transform memory using a function.
        
        Args:
            memory: Memory to transform
            transform_func: Function to transform messages
            **kwargs: Additional arguments for transform function
            
        Returns:
            Transformed memory instance
        """
        # Create new memory of same type
        transformed = type(memory)()
        
        # Transform messages
        for msg in memory.messages:
            transformed_msg = transform_func(msg, **kwargs)
            if transformed_msg:
                await transformed.add_message(
                    transformed_msg["message"],
                    transformed_msg["metadata"]
                )
                
        return transformed

    @staticmethod
    async def analyze_memory(
        memory: BaseMemory
    ) -> Dict[str, Any]:
        """
        Analyze memory contents.
        
        Args:
            memory: Memory to analyze
            
        Returns:
            Analysis results
        """
        if not memory.messages:
            return {
                "message_count": 0,
                "roles": {},
                "average_length": 0,
                "time_span": None
            }
            
        # Calculate statistics
        roles = {}
        total_length = 0
        timestamps = []
        
        for msg in memory.messages:
            # Count roles
            role = msg["message"].get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            
            # Calculate length
            content = msg["message"].get("content", "")
            total_length += len(content)
            
            # Track timestamps
            timestamps.append(msg["timestamp"])
            
        # Calculate time span
        if timestamps:
            time_span = max(timestamps) - min(timestamps)
        else:
            time_span = None
            
        return {
            "message_count": len(memory.messages),
            "roles": roles,
            "average_length": total_length / len(memory.messages),
            "time_span": time_span,
            "metadata_keys": list(memory.metadata.keys())
        }

    @staticmethod
    async def compare_memories(
        memory1: BaseMemory,
        memory2: BaseMemory
    ) -> Dict[str, Any]:
        """
        Compare two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Comparison results
        """
        # Get basic stats
        stats1 = await MemoryUtils.analyze_memory(memory1)
        stats2 = await MemoryUtils.analyze_memory(memory2)
        
        # Calculate overlap
        content1 = {
            msg["message"].get("content", "")
            for msg in memory1.messages
        }
        content2 = {
            msg["message"].get("content", "")
            for msg in memory2.messages
        }
        
        overlap = len(content1.intersection(content2))
        total = len(content1.union(content2))
        
        return {
            "memory1_stats": stats1,
            "memory2_stats": stats2,
            "content_overlap": overlap / total if total > 0 else 0.0,
            "message_count_diff": abs(
                stats1["message_count"] - stats2["message_count"]
            ),
            "role_diff": {
                role: abs(
                    stats1["roles"].get(role, 0) -
                    stats2["roles"].get(role, 0)
                )
                for role in set(
                    stats1["roles"].keys()
                ).union(
                    stats2["roles"].keys()
                )
            }
        } 