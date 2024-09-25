import numpy as np
from typing import Callable

class SimplePerceptron:
    def __init__(self, num_inputs, learning_rate = 0.01, threshold: float = 0.0, epsilon: float = 1e-5, debug = False) -> None:
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate
        self.debug = debug
        self.__THRESHOLD = threshold
        self.__EPSILON = epsilon

    def linear(self, inputs):
        # Add a bias constant term to the inputs
        inputs_with_bias = self.add_bias(inputs)
        # @ is the matrix multiplication operator in Python
        # equivalent to array.dot(other_array)
        Z = inputs_with_bias @ self.weights
        return Z
    
    def add_bias(self, inputs):
        if inputs.ndim == 1:
            return np.hstack(([1], inputs))
        else:
            return np.hstack((np.ones((inputs.shape[0], 1)), inputs))

    # Define the step function.
    def step_fn(self, z: np.ndarray) -> np.ndarray:
        return np.where(z >= self.__THRESHOLD, 1, -1)
         
    def predict(self, inputs: np.ndarray, activation_fn: Callable[[np.ndarray], np.ndarray] = None) -> np.ndarray:
        Z = self.linear(inputs)
        if activation_fn is None:
            activation_fn = self.step_fn
        return activation_fn(Z)
     
    # Define the Loss function
    def loss(self, prediction, target):
        loss = (target-prediction)
        return loss
     
    #Define training
    def train(self, inputs, target) -> float:
        prediction = self.predict(inputs)
        error = self.loss(prediction, target)
        inputs_with_bias = self.add_bias(inputs)
        self.weights += self.learning_rate * error * inputs_with_bias
        return error
         
    # Fit the model
    def fit(self, X, y, num_epochs):
        total_error = 0
        if self.debug:
            print(f'Initial Weights: {self.weights}')
        for epoch in range(num_epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                error = self.train(inputs, target)
                total_error += np.abs(error)
            if self.debug:
                print(f'Epoch {epoch+1}/{num_epochs}, Total Error: {total_error}, Weights: {self.weights}')
            if total_error < self.__EPSILON:
                print(f'Converged after {epoch+1} epochs.')
                break
            else:
                print(f'Did not converge after {num_epochs} epochs.')