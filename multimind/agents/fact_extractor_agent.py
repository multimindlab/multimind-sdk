import re
from typing import List, Tuple
from multimind.memory.graph_memory import GraphMemoryAgent

class FactExtractorAgent:
    """
    Parses LLM outputs or text into (subject, predicate, object) triples and adds them to GraphMemoryAgent.
    For demo, uses a simple regex for extraction.
    """
    def __init__(self, memory_agent: GraphMemoryAgent):
        self.memory_agent = memory_agent

    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract triples from text. For demo, expects lines like 'A <predicate> B'.
        """
        triples = []
        pattern = re.compile(r'([\w\s]+)\s+(\w+)\s+([\w\s]+)')
        for line in text.splitlines():
            match = pattern.match(line.strip())
            if match:
                subj, pred, obj = match.groups()
                triples.append((subj.strip(), pred.strip(), obj.strip()))
        return triples

    def add_facts_from_text(self, text: str) -> int:
        """
        Extracts and adds all found triples to memory. Returns number added.
        """
        triples = self.extract_facts(text)
        count = 0
        for subj, pred, obj in triples:
            if self.memory_agent.add_fact(subj, pred, obj):
                count += 1
        return count 