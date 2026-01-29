import numpy as np
from typing import List

def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    return float(np.linalg.norm(v1 - v2))

def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def cosine_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine distance between two vectors
    """
    return 1 - cosine_similarity(vector1, vector2)

def calculate_distance(vector1: List[float], vector2: List[float], 
                      metric: str = 'cosine') -> float:
    """
    Calculate distance between two vectors using specified metric
    """
    if metric == 'euclidean':
        return euclidean_distance(vector1, vector2)
    elif metric == 'cosine':
        return cosine_distance(vector1, vector2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
