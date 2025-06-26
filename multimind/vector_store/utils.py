import numpy as np
from typing import List, Dict, Any

def normalize_vector(vec: List[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    return float(np.linalg.norm(np.array(vec1, dtype=np.float32) - np.array(vec2, dtype=np.float32)))

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def filter_by_metadata(items: List[Dict[str, Any]], filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    def match(meta):
        return all(meta.get(k) == v for k, v in filter_criteria.items())
    return [item for item in items if match(item.get('metadata', {}))] 