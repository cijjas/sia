import numpy as np


class WeightInitializer:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def initialize_weights(topology, activation_function):
        """
        Initialize weights based on the topology and activation function.
        """
        weights = []
        for x, y in zip(topology[:-1], topology[1:]):
            if activation_function == "relu":
                # He Initialization: N(0, sqrt(2 / fan_in))
                std_dev = np.sqrt(np.divide(2, x))
                weight_matrix = np.multiply(np.random.randn(y, x), std_dev)
            elif activation_function in ["sigmoid", "tanh"]:
                # Xavier Initialization: N(0, sqrt(1 / fan_in))
                std_dev = np.sqrt(np.divide(1, x))
                weight_matrix = np.multiply(np.random.randn(y, x), std_dev)
            else:
                # Default Initialization: N(0, 0.01)
                weight_matrix = np.multiply(np.random.randn(y, x), 0.01)
            weights.append(weight_matrix)
        return weights

    @staticmethod
    def initialize_biases(topology):
        """
        Initialize biases as zeros for the given topology.
        """
        return [np.zeros((y, 1)) for y in topology[1:]]
