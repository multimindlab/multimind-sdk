"""
Utility functions for memory management.
"""

from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
import json
from pathlib import Path
import pickle
from .base import BaseMemory
import numpy as np

class AdaptiveThreshold:
    """
    Adaptive threshold for similarity-based filtering.
    Tracks recent scores and adapts threshold based on distribution or feedback.
    Usage:
        threshold = AdaptiveThreshold(initial=0.8)
        for score in scores:
            threshold.update(score, feedback=1.0)  # feedback=1.0 for good, 0.0 for bad
        current = threshold.value
    """
    def __init__(self, initial: float = 0.8, window: int = 50, min_val: float = 0.5, max_val: float = 0.95):
        self.value = initial
        self.window = window
        self.scores = []
        self.feedback = []
        self.min_val = min_val
        self.max_val = max_val
    def update(self, score: float, feedback: float = None):
        self.scores.append(score)
        if feedback is not None:
            self.feedback.append(feedback)
        if len(self.scores) > self.window:
            self.scores.pop(0)
        if len(self.feedback) > self.window:
            self.feedback.pop(0)
        # Adapt threshold: e.g., set to mean - std, or based on feedback
        if self.feedback:
            # If recent feedback is low, lower threshold; if high, raise
            avg_feedback = np.mean(self.feedback[-self.window:])
            if avg_feedback < 0.5:
                self.value = max(self.min_val, self.value - 0.01)
            elif avg_feedback > 0.8:
                self.value = min(self.max_val, self.value + 0.01)
        else:
            # Use score distribution
            mean = np.mean(self.scores[-self.window:])
            std = np.std(self.scores[-self.window:])
            self.value = np.clip(mean - std, self.min_val, self.max_val)

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
        strategy: str = "append",  # append, interleave, smart
        similarity_func: callable = None,
        adaptive_threshold: 'AdaptiveThreshold' = None,
        llm: Any = None
    ) -> BaseMemory:
        """
        Merge multiple memories into one.
        Args:
            memories: List of memories to merge
            strategy: Merge strategy
            similarity_func: Function to compute similarity (for smart merge; defaults to sentence_bert_similarity)
            adaptive_threshold: AdaptiveThreshold instance (for smart merge)
            llm: Optional LLM for LLM-based similarity
        Returns:
            Merged memory instance
        """
        if not memories:
            raise ValueError("No memories to merge")
        merged = type(memories[0])()
        if strategy == "append":
            for memory in memories:
                for msg in memory.messages:
                    await merged.add_message(
                        msg["message"],
                        msg["metadata"]
                    )
        elif strategy == "interleave":
            all_messages = []
            for memory in memories:
                all_messages.extend(memory.messages)
            all_messages.sort(
                key=lambda x: x["timestamp"]
            )
            for msg in all_messages:
                await merged.add_message(
                    msg["message"],
                    msg["metadata"]
                )
        else:  # smart
            seen_content = []
            if similarity_func is None:
                similarity_func = MemoryUtils.sentence_bert_similarity
            for memory in memories:
                for msg in memory.messages:
                    content = msg["message"].get("content", "")
                    add = True
                    for seen in seen_content:
                        if llm and similarity_func == MemoryUtils.llm_similarity:
                            sim = await similarity_func(content, seen, llm=llm)
                        else:
                            sim = similarity_func(content, seen)
                        if adaptive_threshold:
                            adaptive_threshold.update(sim)
                            if sim > adaptive_threshold.value:
                                add = False
                                break
                        elif sim > 0.85:  # default threshold for semantic deduplication
                            add = False
                            break
                    if add:
                        await merged.add_message(
                            msg["message"],
                            msg["metadata"]
                        )
                        seen_content.append(content)
        return merged

    @staticmethod
    async def filter_memory(
        memory: BaseMemory,
        filter_func: callable = None,
        similarity_func: callable = None,
        adaptive_threshold: 'AdaptiveThreshold' = None,
        **kwargs
    ) -> BaseMemory:
        """
        Filter memory based on a function or similarity threshold.
        Args:
            memory: Memory to filter
            filter_func: Function to filter messages
            similarity_func: Function to compute similarity (optional)
            adaptive_threshold: AdaptiveThreshold instance (optional)
            **kwargs: Additional arguments for filter function
        Returns:
            Filtered memory instance
        """
        filtered = type(memory)()
        for msg in memory.messages:
            keep = True
            if filter_func:
                keep = filter_func(msg, **kwargs)
            elif similarity_func and adaptive_threshold:
                # Compare to previous messages
                for prev in filtered.messages:
                    sim = similarity_func(
                        msg["message"].get("content", ""),
                        prev["message"].get("content", "")
                    )
                    adaptive_threshold.update(sim)
                    if sim > adaptive_threshold.value:
                        keep = False
                        break
            if keep:
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

    @staticmethod
    def bertscore_similarity(a: str, b: str) -> float:
        """Compute BERTScore similarity between two texts (requires bert-score)."""
        try:
            from bert_score import score
            P, R, F1 = score([a], [b], lang="en", verbose=False)
            return float(F1[0])
        except ImportError:
            raise ImportError("bertscore is not installed. Run 'pip install bert-score'.")

    @staticmethod
    def sentence_bert_similarity(a: str, b: str) -> float:
        """Compute Sentence-BERT cosine similarity (requires sentence-transformers)."""
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(a, convert_to_tensor=True)
            emb2 = model.encode(b, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(emb1, emb2).item())
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Run 'pip install sentence-transformers'.")

    @staticmethod
    async def llm_similarity(a: str, b: str, llm=None) -> float:
        """Compute similarity using an LLM (requires llm with async generate)."""
        if llm is None:
            raise ValueError("LLM instance must be provided for LLM-based similarity.")
        prompt = f"On a scale from 0 to 1, how similar are the following two texts?\nText 1: {a}\nText 2: {b}\nSimilarity (0-1):"
        response = await llm.generate(prompt)
        try:
            return float(response.strip())
        except Exception:
            return 0.0 