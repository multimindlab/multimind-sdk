import re
from typing import List, Tuple, Callable, Optional

class FactExtractorAgent:
    """
    Parses LLM outputs or text into (subject, predicate, object) triples and adds them to a memory agent (e.g., GraphMemoryAgent).
    Supports custom extraction functions, regex, or LLM-based extraction.
    Modular and developer-friendly for use in pipelines or reflexive loops.
    """
    def __init__(self, memory_agent=None, extract_fn: Optional[Callable[[str], List[Tuple[str, str, str]]]] = None):
        """
        memory_agent: Optional memory agent (e.g., GraphMemoryAgent) to store extracted facts.
        extract_fn: Optional custom function to extract triples. If None, uses default regex-based extraction.
        The function signature is (text) -> list of (subject, predicate, object).
        """
        self.memory_agent = memory_agent
        self.extract_fn = extract_fn or self.default_extract

    def default_extract(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Default extraction: expects lines like 'A <predicate> B'.
        """
        triples = []
        pattern = re.compile(r'([\w\s]+)\s+(\w+)\s+([\w\s]+)')
        for line in text.splitlines():
            match = pattern.match(line.strip())
            if match:
                subj, pred, obj = match.groups()
                triples.append((subj.strip(), pred.strip(), obj.strip()))
        return triples

    def extract_and_store(self, text: str) -> int:
        """
        Extracts triples from text and adds them to memory. Returns number added.
        """
        triples = self.extract_fn(text)
        count = 0
        if self.memory_agent and hasattr(self.memory_agent, 'add_fact'):
            for subj, pred, obj in triples:
                if self.memory_agent.add_fact(subj, pred, obj):
                    count += 1
        return count 