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
        return 1.0 / (1.0 + np.exp(-self.beta * z))

    def _sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        a = self._sigmoid(z)
        return self.beta * a * (1 - a)

    def _tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * z)

    def _tanh_prime(self, z: np.ndarray) -> np.ndarray:
        return self.beta * (1 - np.tanh(self.beta * z) ** 2)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _relu_prime(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.0)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _softmax_prime(self, z: np.ndarray) -> np.ndarray:
        # La derivada del softmax es más compleja y normalmente se utiliza en combinación con la función de pérdida de entropía cruzada.
        # Para fines prácticos, esta implementación sirve como un placeholder.
        s = self._softmax(z)
        return s * (1 - s)

    def __str__(self) -> str:
        return f"ActivationFunction(method='{self.method}', beta={self.beta})"
