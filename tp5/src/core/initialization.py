import numpy as np


class WeightInitializer:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def initialize_weights(topology, activation_function):
        weights = []
        for x, y in zip(topology[:-1], topology[1:]):
            if activation_function == "relu":
                # He Initialization
                weight_matrix = np.random.randn(y, x) * np.sqrt(2 / x)
            elif activation_function in ["sigmoid", "tanh"]:
                # Xavier Initialization
                weight_matrix = np.random.randn(y, x) * np.sqrt(1 / x)
            else:
                # Default Initialization
                weight_matrix = np.random.randn(y, x) * 0.01
            weights.append(weight_matrix)
        return weights

    @staticmethod
    def initialize_biases(topology):
        return [np.zeros((y, 1)) for y in topology[1:]]
