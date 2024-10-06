import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score

from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):
    
    def __init__(self, seed, sizes, activation_function: ActivationFunction, optimizer: Optimizer, weights=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_layers = len(sizes)
        self.sigma = activation_function
        self.optimizer = optimizer
        self.sizes = sizes

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(y, x + 1) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for w in self.weights:
            a_aug = np.vstack([a, [[1]]])  # Append 1 to activation for bias
            a = self.sigma.activation(np.dot(w, a_aug))
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f"Epoch {j}: {accuracy} / {n_test}")
            else:
                print(f"Epoch {j} complete")

            # Calculate loss and gradient magnitudes

    

   

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights by applying gradient descent using backpropagation."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update weights using the optimizer
        self.weights = self.optimizer.update_parameters(self.weights, nabla_w)

    def backprop(self, x, y):
        """Return the gradient for the cost function."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x]  # Store activations layer by layer
        zs = []  # Store z vectors layer by layer
        for w in self.weights:
            a_aug = np.vstack([activation, [[1]]])  # Append 1 for bias
            z = np.dot(w, a_aug)
            zs.append(z)
            activation = self.sigma.activation(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.sigma.activation_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, np.vstack([activations[-2], [[1]]]).T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigma.activation_prime(z)
            w_no_bias = self.weights[-l + 1][:, :-1]  # Exclude bias from weights
            delta = np.dot(w_no_bias.T, delta) * sp
            nabla_w[-l] = np.dot(delta, np.vstack([activations[-l - 1], [[1]]]).T)
        return nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct result."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives."""
        return (output_activations - y)
    
    def save_model(self, filename):
        weights_array = np.array(self.weights, dtype=object)
        sizes_array = np.array(self.sizes, dtype=object)
        # Save weights and topology to a npz file
        np.savez_compressed(filename, weights=weights_array, sizes=sizes_array)

    @classmethod
    def load_model(cls, filename, activation_function, optimizer, seed=None):
        # Load weights from a npz file
        data = np.load(filename, allow_pickle=True)
        weights = data['weights']
        sizes = data['sizes'].tolist()
        # Convert weights back to lists
        weights = weights.tolist()
        model = cls(
            seed=seed,
            sizes=sizes,
            activation_function=activation_function,
            optimizer=optimizer,
            weights=weights
        )
        return model
