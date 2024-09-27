import numpy as np
from typing import Callable
from perceptron_base import PerceptronBase

class PerceptronSimple(PerceptronBase):
    
    def __init__(self, num_inputs,  weights=None, learning_rate = 0.01, threshold: float = 0.0, epsilon: float = 1e-5) -> None:
        super().__init__(num_inputs, weights, learning_rate, threshold, epsilon)
        
    def compute_activation(self, h_mu):
        return 1 if h_mu > 0 else -1
    
    def gradient(self, h_mu):
        return 1