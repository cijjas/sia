import numpy as np


class ActivationFunction:
    def __init__(self, method: str, beta: float = 1.0):
        self.method = method.lower()
        self.beta = beta
        self.activation, self.activation_prime = self._get_activation_functions()

    def _get_activation_functions(self):
        functions = {
            "sigmoid": (self._sigmoid, self._sigmoid_prime),
            "tanh": (self._tanh, self._tanh_prime),
            "relu": (self._relu, self._relu_prime),
            "softmax": (self._softmax, self._softmax_prime),
        }
        if self.method not in functions:
            raise ValueError(f"Unknown activation function: {self.method}")
        return functions[self.method]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: 1 / (1 + exp(-beta * z))
        """
        return np.divide(1.0, np.add(1.0, np.exp(np.multiply(-self.beta, z))))

    def _sigmoid_prime(self, a: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid using activation output a: beta * a * (1 - a)
        """
        return np.multiply(self.beta, np.multiply(a, np.subtract(1, a)))

    def _tanh(self, z: np.ndarray) -> np.ndarray:
        """
        Tanh activation: tanh(beta * z)
        """
        return np.tanh(np.multiply(self.beta, z))

    def _tanh_prime(self, a: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh using activation output a: beta * (1 - a^2)
        """
        return np.multiply(self.beta, np.subtract(1, np.square(a)))

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """
        ReLU activation: max(0, z)
        """
        return np.maximum(0, z)

    def _relu_prime(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: 1 if z > 0 else 0
        """
        return np.where(z > 0, 1.0, 0.0)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax activation: exp(z) / sum(exp(z)) along the last axis
        """
        exp_z = np.exp(np.subtract(z, np.max(z, axis=-1, keepdims=True)))
        return np.divide(exp_z, np.sum(exp_z, axis=-1, keepdims=True))

    def _softmax_prime(self, a: np.ndarray) -> np.ndarray:
        """
        Placeholder for derivative of softmax.
        Note: Typically used in combination with cross-entropy loss.
        """
        raise NotImplementedError(
            "The derivative of softmax is complex and context-specific. Use cross-entropy for simplicity."
        )

    def __str__(self) -> str:
        return f"ActivationFunction(method='{self.method}', beta={self.beta}')"
