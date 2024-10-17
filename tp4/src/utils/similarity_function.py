import numpy as np

EUCLIDEAN_DISTANCE = "euclidean"
COSINE_SIMILARITY = "cosine"
EXPONENTIAL_SIMILARITY = "exponential"


# euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


# cosine similarity
def cosine_similarity(a, b):
    # Compute dot product between each row of 'a' and 'b'
    dot_product = np.dot(a, b)
    # Compute the norm of each row in 'a'
    norm_a = np.linalg.norm(a, axis=-1)
    # Compute the norm of 'b'
    norm_b = np.linalg.norm(b)
    # Return the cosine similarity for each row in 'a'
    return dot_product / (norm_a * norm_b)


# exponential similarity
def exponential_similarity(a, b):
    # Compute the squared Euclidean distance between each row of 'a' and 'b'
    squared_distance = np.linalg.norm(a - b, axis=-1) ** 2
    # Apply the exponential decay
    return np.exp(-squared_distance)


def str_to_similarity_function(similarity_function) -> callable:
    if similarity_function == EUCLIDEAN_DISTANCE:
        return euclidean_distance
    elif similarity_function == COSINE_SIMILARITY:
        return cosine_similarity
    elif similarity_function == EXPONENTIAL_SIMILARITY:
        return exponential_similarity
    else:
        raise ValueError(f"Invalid similarity function: {similarity_function}")
