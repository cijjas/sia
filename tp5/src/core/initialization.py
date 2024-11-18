import numpy as np


class WeightInitializer:
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
                raise ValueError(
                    f"Unsupported activation function: {activation_function}. "
                    "Supported functions: 'relu', 'sigmoid', 'tanh'."
                )
            weights.append(weight_matrix)
        return weights

    @staticmethod
    def initialize_biases(topology):

        return [np.zeros((y, 1)) for y in topology[1:]]
