import numpy as np

EUCLIDEAN_DISTANCE = 'euclidean'
COSINE_SIMILARITY = 'cosine'
EXPONENTIAL_SIMILARITY = 'exponential'

# euclidean distance
def euclidean_distance(a, b, axis=-1):
    return np.linalg.norm(a - b, axis=axis)


# cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# exponential similarity
def exponential_similarity(a: np.ndarray, b: np.ndarray, sigma: float=0.01) -> float:
    return np.exp(-np.linalg.norm(a - b) / (2 * sigma**2))


def str_to_similarity_function(similarity_function) -> callable:
    if similarity_function == EUCLIDEAN_DISTANCE:
        return euclidean_distance
    elif similarity_function == COSINE_SIMILARITY:
        return cosine_similarity
    elif similarity_function == EXPONENTIAL_SIMILARITY:
        return exponential_similarity
    else:
        raise ValueError(f'Invalid similarity function: {similarity_function}')


