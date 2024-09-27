import numpy as np
from typing import Callable


## Fue copiado exactamente de las presentaciones.
## TODO falta agregar otro metodo de termination? no se no mire esa parte todavia
class PerceptronBase:
    def __init__(self, num_inputs,  weights=None, learning_rate = 0.01, threshold: float = 0.0, epsilon: float = 1e-5) -> None:
        self.weights = np.random.normal(0, 0.01, num_inputs +1) if weights is None else weights
        self.learning_rate = learning_rate
        self.weights_history = [self.weights.copy()]
        self.epsilon = epsilon


    def compute_activation(self, h_mu):
        pass

    def gradient(self, h_mu):
        pass
   
    # exactamente copiado de lo que nos dio la catedra
    # NO FIXME intentaría no irnos de esta idea
    def fit(self, X, y, num_epochs):
        converged = False
        for epoch in range(num_epochs):
            total_error = 0
            for mu in range(len(X)): # for each training example mu in the dataset
                
                # Add the bias to the input
                X_mu = np.concatenate(([1], X[mu]))

                # 1. calculate the weighted sum 
                h_mu = np.dot(self.weights, X_mu)

                # 2. Compute the activation given by theta(h_mu)
                O_h_mu = self.compute_activation(h_mu)

                
                # 3. update the weights and bias
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * (y[mu] - O_h_mu) * X_mu[i] * self.gradient(h_mu)

                # 4. calculate the perceptron error
                error = y[mu] - O_h_mu
                total_error += abs(error)

            self.weights_history.append(self.weights.copy())
            if total_error < self.epsilon:
                converged = True
                break

        if not converged:
            print(f"Perceptron did not converge in {num_epochs} epochs")
        else:
            print(f"Perceptron converged in {epoch+1} epochs")
        
        return converged

                
          