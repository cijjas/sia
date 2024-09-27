import numpy as np
from typing import Callable
from perceptron.perceptron_base import PerceptronBase

class PerceptronNonLinear(PerceptronBase):

    # esta es la funcion
    non_linear_functions = {
        'logistic': lambda x, beta: 1 / (1 + np.exp(-2*beta*x)),
        'tanh': lambda x, beta: np.tanh(beta*x),
        'relu': lambda x, beta: np.maximum(0, x)
    }

    # esta es la derivada
    non_linear_gradients = {
        'logistic': lambda x, beta: 2 * beta * x * (1 - x),
        'tanh': lambda x, beta: beta * (1 - x**2),
        'relu': lambda x, beta: 1 if x > 0 else 0
    }


    def __init__(self, num_inputs,  weights=None, learning_rate = 0.01, threshold: float = 0.0, epsilon: float = 1e-5, non_linear_fn ="logistic", beta = 1) -> None:
        self.fn = self.non_linear_functions[non_linear_fn]
        self.gr = self.non_linear_gradients[self.non_linear_fn]
        self.beta = beta
        super().__init__(num_inputs, weights, learning_rate, threshold, epsilon)



    def compute_activation(self, h_mu):
        return self.fn(h_mu, self.beta)
    

    
    def gradient(self, h_mu):
        return self.gr(h_mu, self.beta)