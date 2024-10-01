import numpy as np
from typing import Callable

class PerceptronBase:
    def __init__(self,seed, num_features,  weights=None, learning_rate = 0.01, epsilon: float = 1e-5) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.normal(0, 0.01, num_features +1) if weights is None else weights
        self.learning_rate = learning_rate
        self.weights_history = [self.weights.copy()]
        self.loss_history = []
        self.epsilon = epsilon


    def compute_activation(self, h_mu):
        pass

    def compute_gradient(self, h_mu):
        pass

    def compute_error(self, expected, actual):
        pass
   
    
    def predict(self, X):
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        
        h = X_bias @ self.weights

        O_h = self.compute_activation(h)

        return O_h
    
    
    # exactamente copiado de lo que nos dio la catedra
    def train(self, X, y, num_epochs):
        converged = False
        n = X.shape[0] # number of training examples
        for epoch in range(num_epochs):
            expected, actual = [], []
            for mu in range(n): # for each training example mu in the dataset

                # Add the bias to the feature vector
                X_mu = np.concatenate(([1], X[mu]))

                # 1. calculate the weighted sum 
                h_mu = self.weights @ X_mu
                
                # 2. Compute the activation given by theta(h_mu)
                O_h_mu = self.compute_activation(h_mu)
                
                # 3. update the weights and bias
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * (y[mu] - O_h_mu) * X_mu[i] * self.compute_gradient(h_mu)
                
                expected.append(y[mu])
                actual.append(O_h_mu)

            # 4. calculate the perceptron error
            
            error = self.compute_error(expected, actual)
            #print(f"------------------------------ Epoch {epoch+1}, Error: {error} ----------------------------------------")
            self.weights_history.append(self.weights.copy())
            self.loss_history.append(error)
            if error < self.epsilon:
                converged = True
                break

        if not converged:
            print(f"Perceptron did not converge in {num_epochs} epochs")
        else:
            print(f"Perceptron converged in {epoch+1} epochs")
        
        return converged

                
          