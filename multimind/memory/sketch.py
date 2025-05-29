"""
Compressed Sketch-Based Memory implementation using probabilistic data structures.
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import mmh3  # MurmurHash3 for hashing
from .base import BaseMemory

class CountMinSketch:
    """Count-Min Sketch implementation for frequency estimation."""
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.counts = np.zeros((depth, width), dtype=np.int32)
        self.seeds = np.random.randint(0, 2**32, depth)

    def add(self, key: str, count: int = 1) -> None:
        """Add an element to the sketch."""
        for i in range(self.depth):
            hash_val = mmh3.hash(key, self.seeds[i]) % self.width
            self.counts[i, hash_val] += count

    def estimate(self, key: str) -> int:
        """Estimate the frequency of an element."""
        return min(
            self.counts[i, mmh3.hash(key, self.seeds[i]) % self.width]
            for i in range(self.depth)
        )

class BloomFilter:
    """Bloom Filter implementation for membership testing."""
    def __init__(self, size: int = 10000, num_hashes: int = 7):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = np.zeros(size, dtype=bool)
        self.seeds = np.random.randint(0, 2**32, num_hashes)

    def add(self, key: str) -> None:
        """Add an element to the filter."""
        for i in range(self.num_hashes):
            hash_val = mmh3.hash(key, self.seeds[i]) % self.size
            self.bits[hash_val] = True

    def contains(self, key: str) -> bool:
        """Check if an element is in the filter."""
        return all(
            self.bits[mmh3.hash(key, self.seeds[i]) % self.size]
            for i in range(self.num_hashes)
        )

class HyperLogLog:
    """HyperLogLog implementation for cardinality estimation."""
    def __init__(self, precision: int = 4):
        self.precision = precision
        self.m = 1 << precision
        self.M = np.zeros(self.m, dtype=np.int8)
        self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def add(self, key: str) -> None:
        """Add an element to the counter."""
        x = mmh3.hash(key)
        j = x & (self.m - 1)
        w = x >> self.precision
        self.M[j] = max(self.M[j], self._count_leading_zeros(w))

    def estimate(self) -> float:
        """Estimate the cardinality."""
        E = self.alpha * self.m * self.m / np.sum(2.0 ** -self.M)
        if E <= 2.5 * self.m:
            V = np.sum(self.M == 0)
            if V > 0:
                E = self.m * np.log(self.m / V)
        return E

    def _count_leading_zeros(self, x: int) -> int:
        """Count leading zeros in binary representation."""
        return 32 - len(bin(x)[2:])

class SketchMemory(BaseMemory):
    """Memory implementation using compressed sketches."""

    def __init__(
        self,
        sketch_width: int = 1000,
        sketch_depth: int = 5,
        bloom_size: int = 10000,
        bloom_hashes: int = 7,
        hll_precision: int = 4,
        **kwargs
    ):
        """Initialize sketch memory."""
        super().__init__(**kwargs)
        
        # Initialize sketches
        self.frequency_sketch = CountMinSketch(sketch_width, sketch_depth)
        self.membership_filter = BloomFilter(bloom_size, bloom_hashes)
        self.cardinality_counter = HyperLogLog(hll_precision)
        
        # Memory tracking
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, List[datetime]] = defaultdict(list)
        self.last_access: Dict[str, datetime] = {}
        
        # Statistics
        self.total_adds = 0
        self.total_queries = 0
        self.false_positives = 0

    async def add_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new memory with sketch tracking."""
        # Create memory entry
        memory = {
            'id': memory_id,
            'content': content,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Update sketches
        self.frequency_sketch.add(memory_id)
        self.membership_filter.add(memory_id)
        self.cardinality_counter.add(memory_id)
        
        # Update statistics
        self.total_adds += 1

    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        self.total_queries += 1
        
        # Check membership filter first
        if not self.membership_filter.contains(memory_id):
            return None
        
        # Get memory if it exists
        memory = self.memories.get(memory_id)
        if memory:
            # Update access tracking
            memory['access_count'] += 1
            memory['last_accessed'] = datetime.now()
            self.access_times[memory_id].append(datetime.now())
            self.last_access[memory_id] = datetime.now()
            return memory
        else:
            # False positive in bloom filter
            self.false_positives += 1
            return None

    async def estimate_frequency(self, memory_id: str) -> int:
        """Estimate how many times a memory has been accessed."""
        return self.frequency_sketch.estimate(memory_id)

    async def estimate_cardinality(self) -> float:
        """Estimate the total number of unique memories."""
        return self.cardinality_counter.estimate()

    async def get_access_pattern(
        self,
        memory_id: str,
        time_window: Optional[timedelta] = None
    ) -> List[datetime]:
        """Get access pattern for a memory."""
        if memory_id not in self.access_times:
            return []
            
        times = self.access_times[memory_id]
        if time_window:
            cutoff = datetime.now() - time_window
            times = [t for t in times if t >= cutoff]
        return times

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_memories': len(self.memories),
            'estimated_cardinality': await self.estimate_cardinality(),
            'total_adds': self.total_adds,
            'total_queries': self.total_queries,
            'false_positive_rate': self.false_positives / self.total_queries if self.total_queries > 0 else 0.0,
            'avg_access_count': np.mean([
                len(times) for times in self.access_times.values()
            ]) if self.access_times else 0.0
        }

    async def remove_memory(self, memory_id: str) -> None:
        """Remove a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            if memory_id in self.access_times:
                del self.access_times[memory_id]
            if memory_id in self.last_access:
                del self.last_access[memory_id] 