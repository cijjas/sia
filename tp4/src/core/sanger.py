import numpy as np
import pandas as pd
from utils.eta_function import constant_eta

class Sanger:

    def __init__(
        self, seed, num_features, num_components, learning_rate=0.01, eta_function=constant_eta, epsilon=1e-5
    ):
        """
        Sanger's rule to find multiple principal components.
        :param seed: Random seed for reproducibility.
        :param num_features: Number of input features.
        :param num_components: Number of principal components to find.
        :param learning_rate: Step size for weight updates.
        :param epsilon: Threshold for convergence.
        """
        if seed is not None:
            np.random.seed(seed)

        self.num_components = num_components
        # Initialize weight matrix for the desired number of components
        self.weights = np.random.normal(0, 1, (num_components, num_features))
        self.learning_rate = learning_rate
        self.eta_function = eta_function
        self.epsilon = epsilon
        self.weights_history = [self.weights.copy()]

    def fit(self, X, epochs):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        n = X.shape[0]  # Number of training examples

        for epoch in range(epochs):
            previous_weights = self.weights.copy()

            for mu in range(n):  # For each training example
                # Compute outputs for all components
                outputs = np.dot(self.weights, X[mu])

                for i in range(self.num_components):

                    # Oja = Sanger's rule for the first component
                    # Sangers rule specially ensures that the components are orthogonal
                    # Correction term ensures that the components are orthogonal
                    # by removing the projection of the previous components
                    correction = np.dot(outputs[: i + 1], self.weights[: i + 1])

                    np.add(
                        self.weights[i],
                        self.learning_rate * outputs[i] * (X[mu] - correction),
                        out=self.weights[i],
                    )

            # Update learning rate
            self.learning_rate = self.eta_function(self.learning_rate, mu, n)

            self.weights_history.append(self.weights.copy())

            weight_change = np.linalg.norm(self.weights - previous_weights)
            if weight_change < self.epsilon:
                print(f"Converged after {epoch+1} epochs.")
                break

        print(f"Sanger's rule completed {epoch+1} epochs.")
        return self.weights
