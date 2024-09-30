import numpy as np
from typing import Callable
from perceptron.perceptron_base import PerceptronBase

class PerceptronSimple(PerceptronBase):
    
    def __init__(self, seed, num_features,  weights=None, learning_rate = 0.01,  epsilon: float = 1e-5) -> None:
        super().__init__(seed, num_features, weights, learning_rate, epsilon)
        
    # perceptron_simple.py

    def compute_activation(self, h):
        # This ensures that if h is an array, the comparison is done element-wise.
        return np.where(h > 0, 1, -1)

    
    def compute_gradient(self, h_mu):
        return 1.0
    
    def compute_error(self, expected, actual):
        error = 0
        for i in range(len(expected)):
            error += abs(expected[i] - actual[i])
        return error