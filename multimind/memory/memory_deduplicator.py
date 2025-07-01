from typing import Any, Callable, Optional

class MemoryDeduplicator:
    """
    Detects and prevents redundant or contradictory facts in the triple store.
    Supports exact, semantic/fuzzy, and contradiction-based deduplication.
    """
    def __init__(self, contradiction_fn: Optional[Callable] = None, similarity_fn: Optional[Callable] = None):
        self.contradiction_fn = contradiction_fn or self.default_contradiction
        self.similarity_fn = similarity_fn or self.default_similarity

    def is_duplicate(self, subject, predicate, obj, triple_store) -> bool:
        """
        Returns True if the triple (subject, predicate, object) already exists (exact or semantic match).
        """
        # Exact match
        matches = triple_store.find_triples(subject, predicate, obj)
        if matches:
            return True
        # Semantic/fuzzy match
        for u, p, o, meta in triple_store.get_triples():
            if self.similarity_fn((subject, predicate, obj), (u, p, o)):
                return True
        return False

    def is_contradictory(self, subject, predicate, obj, triple_store) -> bool:
        """
        Returns True if the triple contradicts any existing triple (using contradiction_fn).
        """
        for u, p, o, meta in triple_store.get_triples():
            if self.contradiction_fn((subject, predicate, obj), (u, p, o)):
                return True
        return False

    def default_similarity(self, t1, t2) -> bool:
        """
        Default: only exact match. Override for semantic/fuzzy matching.
        """
        return t1 == t2

    def default_contradiction(self, t1, t2) -> bool:
        """
        Default: no contradiction. Override for custom logic (e.g., antonyms, negation).
        """
        return False 