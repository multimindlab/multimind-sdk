from typing import Any, Callable, Dict, Optional

class MemoryMergeEngine:
    """
    Uses LLM, rules, or custom logic to merge, reject, or update existing memory entries.
    Supports pluggable merge strategies and developer-friendly API.
    """
    def __init__(self, merge_fn: Optional[Callable] = None, reject_fn: Optional[Callable] = None, update_fn: Optional[Callable] = None):
        self.merge_fn = merge_fn or self.default_merge
        self.reject_fn = reject_fn or self.default_reject
        self.update_fn = update_fn or self.default_update

    def merge(self, entry1: Dict[str, Any], entry2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two memory entries using the merge function.
        """
        return self.merge_fn(entry1, entry2)

    def reject(self, entry1: Dict[str, Any], entry2: Dict[str, Any]) -> bool:
        """
        Decide whether to reject merging two entries (e.g., if contradictory).
        """
        return self.reject_fn(entry1, entry2)

    def update(self, entry: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a memory entry with new information.
        """
        return self.update_fn(entry, updates)

    def default_merge(self, entry1: Dict[str, Any], entry2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default: merge by combining fields, preferring non-null and latest timestamp.
        Override for LLM or advanced logic.
        """
        merged = entry1.copy()
        for k, v in entry2.items():
            if v is not None:
                if k == 'timestamp':
                    merged[k] = max(merged.get(k, v), v)
                else:
                    merged[k] = v
        return merged

    def default_reject(self, entry1: Dict[str, Any], entry2: Dict[str, Any]) -> bool:
        """
        Default: never reject. Override for contradiction or LLM-based logic.
        """
        return False

    def default_update(self, entry: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default: update entry with new fields.
        """
        entry.update(updates)
        return entry 