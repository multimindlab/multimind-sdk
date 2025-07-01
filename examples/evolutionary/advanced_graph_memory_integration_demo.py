from multimind.memory.graph_memory_agent import GraphMemoryAgent
from multimind.memory.memory_deduplicator import MemoryDeduplicator
from multimind.memory.memory_scorer import MemoryScorer
from multimind.memory.memory_merge_engine import MemoryMergeEngine
from datetime import datetime, timedelta

# Custom semantic similarity: treat 'car' and 'automobile' as similar
SIMILAR_WORDS = {('car', 'automobile'), ('automobile', 'car')}
def semantic_similarity(t1, t2):
    s1, p1, o1 = t1
    s2, p2, o2 = t2
    return (s1 == s2 and p1 == p2 and (o1 == o2 or (o1, o2) in SIMILAR_WORDS))

def contradiction_fn(t1, t2):
    # Contradiction if same subject/predicate but object is 'yes' vs 'no'
    s1, p1, o1 = t1
    s2, p2, o2 = t2
    return s1 == s2 and p1 == p2 and {o1, o2} == {'yes', 'no'}

# Custom merge: prefer latest timestamp
def custom_merge(meta1, meta2):
    return meta1 if meta1.get('timestamp', datetime.min) > meta2.get('timestamp', datetime.min) else meta2

def main():
    deduplicator = MemoryDeduplicator(similarity_fn=semantic_similarity, contradiction_fn=contradiction_fn)
    scorer = MemoryScorer(strategy='importance')
    merger = MemoryMergeEngine(merge_fn=custom_merge)
    memory = GraphMemoryAgent(deduplicator=deduplicator, scorer=scorer, merger=merger)

    # Add facts
    print("Adding: ('Alice', 'drives', 'car')")
    memory.add_fact('Alice', 'drives', 'car', importance=0.8, timestamp=datetime.utcnow())
    print("Adding: ('Alice', 'drives', 'automobile') [semantic duplicate]")
    memory.add_fact('Alice', 'drives', 'automobile', importance=0.9, timestamp=datetime.utcnow() + timedelta(seconds=1))
    print("Adding: ('Bob', 'has_license', 'yes')")
    memory.add_fact('Bob', 'has_license', 'yes', importance=0.7, timestamp=datetime.utcnow())
    print("Adding: ('Bob', 'has_license', 'no') [contradiction]")
    added = memory.add_fact('Bob', 'has_license', 'no', importance=0.6, timestamp=datetime.utcnow() + timedelta(seconds=2))
    print("Contradictory add result:", added)

    # Query all facts
    print("\nAll facts:")
    for fact in memory.get_facts():
        print(fact)

    # Timeline query
    since = datetime.utcnow() - timedelta(seconds=2)
    print(f"\nTimeline since {since}:")
    for fact in memory.get_timeline(since):
        print(fact)

if __name__ == '__main__':
    main() 