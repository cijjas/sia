import numpy as np

from .optimizer import Optimizer
from .activation_function import ActivationFunction
from .initialization import WeightInitializer


class MultilayerPerceptron(object):
    def __init__(
        self,
        seed,
        topology,
        activation_function: ActivationFunction,
        optimizer: Optimizer,
        weights=None,
        biases=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.num_layers = len(topology)
        self.sigma = activation_function
        self.optimizer = optimizer
        self.topology = topology

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.weights = WeightInitializer.initialize_weights(
                topology, activation_function.method
            )
            self.biases = WeightInitializer.initialize_biases(topology)

    # Returns the output of the neural network when 'a' is giving as input
    def feedforward(self, a, return_activations=False):
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            a = self.sigma.activation(np.dot(w, a) + b)
            activations.append(a)
        if return_activations:
            return activations
        else:
            return a

    # Trains the neural network with mini batches of the training data for the an amount of epochs
    def fit(self, training_data, epochs, mini_batch_size, patience=10, min_delta=1e-4):
        n = len(training_data)
        pixel_error_history = []

        # Early stopping variables
        best_error = float("inf")  # Best observed error
        wait = 0  # Epochs without improvement

        for epoch in range(epochs):
            # Shuffle training data each epoch
            shuffled_data = training_data[:]
            np.random.shuffle(shuffled_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, mini_batch_size)

            # Calculate pixel error at the end of the epoch
            pixel_error = 0
            for x, y in training_data:
                output = self.feedforward(x)
                # Count incorrect pixels (comparing binarized output to target)
                pixel_error += np.sum((output > 0.5).astype(int) != y.astype(int))
            pixel_error_history.append(pixel_error)

            print(f"Epoch {epoch + 1}/{epochs}, Pixel error: {pixel_error}")

            # Early stopping logic
            if pixel_error < best_error - min_delta:
                best_error = pixel_error
                wait = 0  # Reset wait counter
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Stopping early at epoch {epoch + 1}. Best pixel error: {best_error}"
                    )
                    break

        return pixel_error_history

    # Uses the mini batch to calculate the adjustments to the weights and biases
    def update_weights_and_biases(self, mini_batch, mini_batch_size):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights, self.biases = self.optimizer.update_parameters(
            self.weights, self.biases, nabla_w, nabla_b, mini_batch_size
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
            activations[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.sigma.activation_prime(
                activations[-l]
            )
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_b, nabla_w)

    # Evaluates the accuracy metric for the neural network
    def evaluate(self, test_data, epsilon):
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(abs(x - y) < epsilon) for (x, y) in test_results) / len(
            test_data
        )

    # Derivative of the error function (Squared Error) with respect to the activatio
    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def encode(self, x):
        activation = x
        for b, w in zip(
            self.biases[: self.num_layers // 2], self.weights[: self.num_layers // 2]
        ):
            activation = self.sigma.activation(np.dot(w, activation) + b)
        return activation

    def decode(self, z):
        activation = z
        for b, w in zip(
            self.biases[self.num_layers // 2 :], self.weights[self.num_layers // 2 :]
        ):
            activation = self.sigma.activation(np.dot(w, activation) + b)
        return activation

    ##################################### Save model or load model #########################################################

    def save_model(self, filename):
        weights_array = np.array(self.weights, dtype=object)
        biases_array = np.array(self.biases, dtype=object)
        topology_array = np.array(self.topology, dtype=object)

        np.savez_compressed(
            filename,
            weights=weights_array,
            biases=biases_array,
            topology=topology_array,
        )

    @classmethod
    def load_model(cls, filename, activation_function, optimizer, seed=None):
        data = np.load(filename, allow_pickle=True)
        weights = data["weights"]
        biases = data["biases"]
        topology = data["topology"].tolist()

        weights = weights.tolist()
        biases = biases.tolist()

        model = cls(
            seed=seed,
            topology=topology,
            activation_function=activation_function,
            optimizer=optimizer,
            weights=weights,
            biases=biases,
        )
        return model
