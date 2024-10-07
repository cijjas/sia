import numpy as np
from typing import Optional
from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction


class MultilayerPerceptron(object):
    def __init__(
        self,
        seed,
        sizes,
        activation_function: ActivationFunction,
        optimizer: Optimizer,
        weights=None,
        biases=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.num_layers = len(sizes)
        self.sigma = activation_function
        self.optimizer = optimizer
        self.sizes = sizes

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [
                np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
            ]

    # Returns the output of the neural network when 'a' is giving as input
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigma.activation(np.dot(w, a) + b)
        return a

    # Trains the neural network with mini batches of the training data for the an amount of epochs
    def fit(self, training_data, epochs, mini_batch_size):
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch)

            print(f"Epoch {j} complete")

    # Uses the mini batch to calculate the adjustments to the weights and biases
    def update_weights_and_biases(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights, self.biases = self.optimizer.update_parameters(
            self.weights, self.biases, nabla_w, nabla_b
        )

    # Returns the gradients
    def backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            # Calculate and store preactivations
            z = np.dot(w, activation) + b
            zs.append(z)
            # Calculate and store activations
            activation = self.sigma.activation(z)
            activations.append(activation)

        # Backward pass
        # Special treatment for output layer
        delta = self.cost_derivative(activations[-1], y) * self.sigma.activation_prime(
            zs[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.sigma.activation_prime(
                zs[-l]
            )
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    # Evaluates the accuracy metric for the neural network
    def evaluate(self, test_data, epsilon):
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(abs(x - y) < epsilon) for (x, y) in test_results)

    # Derivative of the error function (Squared Error) with respect to the activatio
    def cost_derivative(self, output_activations, y):
        return output_activations - y

    ##################################### Save model or load model #########################################################

    def save_model(self, filename):
        weights_array = np.array(self.weights, dtype=object)
        biases_array = np.array(self.biases, dtype=object)
        sizes_array = np.array(self.sizes, dtype=object)

        np.savez_compressed(
            filename,
            weights=weights_array,
            biases=biases_array,
            sizes=sizes_array,
        )

    @classmethod
    def load_model(cls, filename, activation_function, optimizer, seed=None):
        data = np.load(filename, allow_pickle=True)
        weights = data["weights"]
        biases = data["biases"]
        sizes = data["sizes"].tolist()

        weights = weights.tolist()
        biases = biases.tolist()

        model = cls(
            seed=seed,
            sizes=sizes,
            activation_function=activation_function,
            optimizer=optimizer,
            weights=weights,
            biases=biases,
        )
        return model
