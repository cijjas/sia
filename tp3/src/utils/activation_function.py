import numpy as np

class ActivationFunction:
    def __init__(self, method: str, beta: float = 1.0):
        self.method = method
        self.beta = beta
        self.activation, self.activation_prime = self.get_activation()

    def get_activation(self) -> tuple:
        if self.method == "sigmoid":
            return self.sigmoid, self.sigmoid_prime
        elif self.method == "tanh":
            return self.tanh, self.tanh_prime
        elif self.method == "relu":
            return self.relu, self.relu_prime
        elif self.method == "softmax":
            return self.softmax, self.softmax_prime
        else:
            raise ValueError(f"Unknown activation function: {self.method}")

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beta * z))

    def sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        sig = self.sigmoid(z)
        return self.beta * sig * (1 - sig)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.beta * z)

    def tanh_prime(self, z: np.ndarray) -> np.ndarray:
        return self.beta * (1 - np.tanh(self.beta * z) ** 2)

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def relu_prime(self, z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1, 0)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        exps = np.exp(z - np.max(z, axis=-1, keepdims=True))  
        return exps / np.sum(exps, axis=-1, keepdims=True) 
    
    def softmax_prime(self, z: np.ndarray) -> np.ndarray:
        return self.softmax(z) 