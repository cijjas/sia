import numpy as np
from typing import Callable
from perceptron.perceptron_base import PerceptronBase

class PerceptronLinear(PerceptronBase):
    
    def __init__(self, seed, num_features, weights=None, learning_rate=0.01, epsilon=1e-5):
        super().__init__(seed, num_features, weights, learning_rate, epsilon)

    def compute_activation(self, h_mu):
        return h_mu
    
    def compute_gradient(self, h_mu):
        return 1.0
    
    def compute_error(self, expected, actual):
        error = 0
        for i in range(len(expected)):
            #string  = "Expected: " + str(expected[i]) + " Actual: " + str(actual[i]) + " Error: " + str((expected[i] - actual[i])**2)
            #print(i, string)
            error += (expected[i] - actual[i])**2

        #print("Error: ", error/len(expected))
        return error/len(expected)
            
