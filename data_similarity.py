import numpy as np
from scipy.spatial.distance import euclidean

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def euclidean_distance(vec1, vec2):
    return euclidean(vec1, vec2)

# Example usage:
set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}
print("Jaccard Similarity:", jaccard_similarity(set1, set2))

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
print("Cosine Similarity:", cosine_similarity(vec1, vec2))

print("Euclidean Distance:", euclidean_distance(vec1, vec2))
