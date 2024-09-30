import numpy as np

class ActivationFunction:
    def __init__(self):
        pass

    def activation(self, z: np.ndarray) -> np.ndarray:
        pass

    def activation_prime(self, z: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(ActivationFunction):
    def activation(self, z: np.ndarray) -> np.ndarray:
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def activation_prime(self, z: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        return self.activation(z)*(1-self.activation(z))

def str_to_activation_function(activation_function_str: str) -> ActivationFunction:
    if activation_function_str == 'sigmoid':
        return Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {activation_function_str}")


