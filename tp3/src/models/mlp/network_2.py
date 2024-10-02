import random
import numpy as np
from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction
from tqdm import tqdm
class MultilayerPerceptron(object):

    def __init__(self, seed, topology, activation_function, optimizer):
        if seed is not None:
            np.random.seed(seed)
        self.num_layers = len(topology)
        self.topology = topology
        self.biases = [np.random.randn(y, 1) for y in topology[1:]]  # Bias vector for each layer
        self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])]  # Weight matrix for each layer
        self.activation_function = activation_function
        self.optimizer = optimizer

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.activation(np.dot(w, a) + b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in tqdm(range(epochs), desc=f"Training MLP Model", colour="green"):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
           
    def update_mini_batch(self, mini_batch, eta):
        x_batch = np.column_stack([x for x, y in mini_batch])
        y_batch = np.column_stack([y for x, y in mini_batch])

        nabla_b, nabla_w = self.backprop(x_batch, y_batch)

        # Update weights and biases using optimizer
        self.weights, self.biases = self.optimizer.update(
            weights=self.weights,
            biases=self.biases,
            grads_w=[nw / len(mini_batch) for nw in nabla_w],
            grads_b=[nb / len(mini_batch) for nb in nabla_b],
            mini_batch_size=len(mini_batch)
        )

    def backprop(self, x, y):
        # Initialize gradients
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        # Forward pass
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activation_function.activation_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function.activation_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def evaluate(self, test_data, epsilon):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        correct = sum(int(np.all(np.abs(output - y) < epsilon)) for (output, y) in test_results)
        return correct

    def cost_derivative(self, output_activations, y):
        return output_activations - y
