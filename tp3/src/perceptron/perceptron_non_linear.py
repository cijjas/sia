import numpy as np
from typing import Callable
from perceptron.perceptron_base import PerceptronBase


def stable_sigmoid(x, beta):
    return 1 / (1 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

def relu(x, beta):
    return np.maximum(0, x)

class PerceptronNonLinear(PerceptronBase):

    non_linear_functions = {
        'sigmoid': stable_sigmoid,
        'tanh': tanh,
        'relu': relu
    }

    # get sigmoid name
    


    # Use correct derivative calculations
    non_linear_gradients = {
        'sigmoid': lambda x, beta: 2 * beta * stable_sigmoid(x, beta) * (1 - stable_sigmoid(x, beta)),
        'tanh': lambda x, beta: beta * (1 - tanh(x, beta)**2),
        'relu': lambda x, beta: np.where(x >= 0, 1, 0)
    }    


    def __init__(self, seed, num_features,  weights=None, learning_rate = 0.01, epsilon: float = 1e-5, non_linear_fn ="sigmoid", beta = 0.9) -> None:
        self.fn_name = non_linear_fn
        self.fn = self.non_linear_functions[non_linear_fn]
        self.gr = self.non_linear_gradients[non_linear_fn]
        self.beta = beta
        super().__init__(seed, num_features, weights, learning_rate, epsilon)


    def compute_activation(self, h_mu):
        return self.fn(h_mu, self.beta)
    
    def compute_gradient(self, h_mu):
        return self.gr(h_mu, self.beta)
    
    def compute_error(self, expected, actual):
        error = 0
        for i in range(len(expected)):
            error += (expected[i] - actual[i])**2
        return error/len(expected)