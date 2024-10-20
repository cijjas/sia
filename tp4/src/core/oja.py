import numpy as np
from typing import Callable
from utils.eta_function import constant_eta

# Algunos modelos de redes neuronales permiten calcular las componentes principales
# en forma iterativa. Uno de estos métodos es la regla de Oja.
# VENTAJA: reduce costo computacional al no tener que calcular la matriz de covarianza
# DESVENTAJA si hay muchas features, interpretarlo en una sola dimensión es complicado.
# se puede perder información.
class Oja:

    def __init__(
        self, seed, num_features, weights=None, learning_rate=0.01, eta_function=constant_eta, epsilon=1e-5
    ):
        if seed is not None:
            np.random.seed(seed)

        self.weights = (
            np.random.normal(0, 1, num_features) if weights is None else weights
        )
        self.learning_rate = learning_rate
        self.eta_function: callable = eta_function
        self.epsilon = epsilon
        self.weights_history = [self.weights.copy()]

    def fit(self, X, epochs):
        # Ignoramos y porque Oja es un método no supervisado
        n = X.shape[0]  # Número de ejemplos de entrenamiento

        for epoch in range(epochs):
            for mu in range(n):  # Para cada ejemplo de entrenamiento

                # 1. Output of wxx^T
                O_h_mu = self.weights @ X[mu]

                # 2. Oja's rule
                np.add(
                    self.weights,
                    self.learning_rate * (O_h_mu * X[mu] - O_h_mu**2 * self.weights),
                    out=self.weights,
                )

            # Actualizar la tasa de aprendizaje
            self.learning_rate = self.eta_function(self.learning_rate, epoch, epochs)

            # Almacenar el historial de pesos después de cada época
            self.weights_history.append(self.weights.copy())

        print(f"Oja rule completed {epochs} epochs.")
        return self.weights
