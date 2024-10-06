import random
import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):

    def __init__(self, seed, topology:list[int], activation_function: ActivationFunction, optimizer:Optimizer, weights: list[np.ndarray] = None, biases: list[np.ndarray] = None):
        """The list ``topology`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.

        The biases and weights for the network are initialized
        randomly, using a Gaussian distribution with mean 0, and
        variance 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers."""

        if seed is not None:
            np.random.seed(seed)

        if weights is not None:
            self.weights = weights
        else:
            self.weights: list[np.ndarray] = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])] # Matches each layer with the next, ignoring the output layer

        if biases is not None:
            self.biases = biases
        else:
            self.biases: list[np.ndarray] = [np.random.randn(y, 1) for y in topology[1:]] # Bias vector of matrixes ignores the input layer

        self.num_layers: int = len(topology)
        self.topology: list[int] = topology
        self.activation_function = activation_function
        self.optimizer = optimizer

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if 'a' is the input."""

        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.activation(np.dot(w, a) + b)
        return a

    # TODO Remove test_data comment, parameter and logic, unnecesarry overhead
    def fit(self, training_data: list[tuple[np.ndarray, np.ndarray]], epochs: int, mini_batch_size: int, eta: float,
            epsilon: float, test_data: Optional[list[tuple[np.ndarray, np.ndarray]]] = None, test_results:list[tuple[int, int]] = None,
            training_results:list[tuple[int,int]]=None) -> None:
        """Train the neural network using mini-batch stochastic
        gradient descent.

        The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.

        If ``test_data`` is provided then the network will be
        evaluated against the test data after each epoch, and
        partial progress printed out. """

        if test_data is not None:
            n_test = len(test_data)
        n: int = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches:list[list[tuple[np.ndarray, np.ndarray]]] = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, eta)

            # Logic for when test_data is provided
            if test_data is not None:
                if test_results is not None:
                    test_result = []
                    self.evaluate(test_data=test_data, epsilon=epsilon, test_results=test_result)
                    test_results.append(test_result)
                else:
                    print("Epoch {0}: {1}".format(j, self.evaluate(test_data=test_data, epsilon=epsilon)))
            else:
                print("Epoch {0} complete".format(j))
            if training_results is not None:
                training_result = []
                self.evaluate(test_data=training_data[:10], epsilon=epsilon, test_results=training_result)
                training_results.append(training_result)
            
    def fit_with_cross_validation(self, training_data: list[tuple[np.ndarray, np.ndarray]], epochs: int, eta: float, mini_batch_size: int,
                                epsilon: float, n_splits: int, test_results: list[tuple[int, int]]) -> None:
        if len(training_data) % n_splits != 0:
            raise ValueError("The number of splits must be a divisor of the number of training samples")

        k_fold_size = len(training_data) // n_splits
        if mini_batch_size > k_fold_size * (n_splits - 1):
            print(f"Warning: mini_batch_size (={mini_batch_size}) is larger than the number of samples used for training \
                  ({n_splits-1} groups of {k_fold_size} samples each, which equals a training set of {k_fold_size*(n_splits-1)} samples)\
                  . (which means the training is done with one big mini batch instead of many smaller ones)")

        if epochs%n_splits != 0:
            epochs = epochs + n_splits - epochs%n_splits
            print(f"Epochs were adjusted to {epochs} to match the number of splits")

        for i in range(0, epochs, n_splits):
            random.shuffle(training_data)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(training_data):
                training_data_cv = [training_data[i] for i in train_index]
                test_data_cv = [training_data[i] for i in test_index]
                mini_batches: list[list[tuple[np.ndarray, np.ndarray]]] = [
                    training_data_cv[k:k + mini_batch_size]
                    for k in range(0, len(training_data_cv), mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_weights_and_biases(mini_batch, eta)

                
                test_result = []
                self.evaluate(test_data=test_data_cv, epsilon=epsilon, test_results=test_result)
                test_results.append(test_result)

            print(f"Epoch {i}")


    def update_weights_and_biases(self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float) -> None:
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        nabla_b: list[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        nabla_w: list[np.ndarray] = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights, self.biases = self.optimizer.update(
            weights=self.weights,
            biases=self.biases,
            grads_w=nabla_w,
            grads_b=nabla_b,
            mini_batch_size=len(mini_batch)
        )

    def backprop(self, input:np.ndarray, expected:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.

        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy
        arrays, similarto ``self.biases`` and ``self.weights``."""

        nabla_b: list[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        nabla_w: list[np.ndarray] = [np.zeros(w.shape) for w in self.weights]

        activation: np.ndarray = input # Represents a^{(L-1)} when calculating a^{(L)}
        activations: list[np.ndarray] = [input] # Stores all activations for later usage in Backpropagation
        zs: list[np.ndarray] = [] # Stores all pre activations for later usage in Backpropagation

        # Feed Foward with storage
        for b, w in zip(self.biases, self.weights):
            # Preactivation calculation and storage for each layer
            z: np.ndarray = np.dot(w, activation) + b
            zs.append(z)
            # Activation calculation and storage for each layer
            activation = self.activation_function.activation(z)
            activations.append(activation)

        # Last layer needs special treatment
        # δ = ∂C/∂a^{(L)} ​⋅ ∂a^{(L)}/∂z^{(L)}
        delta: np.ndarray = self.cost_derivative(activations[-1], expected) * self.activation_function.activation_prime(zs[-1])
        # ∇b = ∂C/∂b = δ because ∂z/∂b = 1
        nabla_b[-1] = delta
        # ∇w = δ ⋅ ∂z/∂w = δ ⋅ activation^{(L-1)}.T
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate the error
        for l in range(2, self.num_layers):
            # δ = ∂C/∂z^{(l)} = ∂C/∂a^{(l)} ​⋅ ∂a^{(l)}/∂z^{(l)}
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activation_function.activation_prime(zs[-l])
            # δ^{(l-1)} = ∂C/∂a^{(l-1)} = δ^{(l)} ⋅ W^{(l)}
            nabla_b[-l] = delta
            # Gradient of the cost with respect to weights: ∂C/∂W^{(l)} = δ^{(l)} ⋅ a^{(l-1)}^T
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the vector of partial derivatives partial C_x
        partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data: list[tuple[np.ndarray, np.ndarray]], epsilon: float, test_results: list[tuple[np.ndarray, np.ndarray]] = None) -> float:
        if self.topology[-1] == 1:
            return self.single_output_evaluation(test_data, epsilon, test_results)
        else:
            return self.multi_output_evaluation(test_data, epsilon, test_results)


    def single_output_evaluation(self, test_data: list[tuple[np.ndarray, np.ndarray]], epsilon: float, test_results: list[tuple[np.ndarray, np.ndarray]] = None) -> float:
        """Return the number of test inputs for which the neural
        network outputs the correct result."""

        # Initialize test_results if it's not provided
        if test_results is None:
            test_results = []

        # Collect predictions and true labels
        for (x, y) in test_data:
            arr = self.feedforward(x)
            test_results.append((arr, y))

        y_true = [true for _, true in test_results]
        print("True labels:", y_true)

        # y_pred rounds the predicted values to the nearest integer
        y_pred = [int(np.round(pred)) for pred, _ in test_results]
        print("Predicted labels:", y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def multi_output_evaluation(self, test_data: list[tuple[np.ndarray, np.ndarray]], epsilon: float, test_results: list[tuple[np.ndarray, np.ndarray]] = None) -> float:
        """Return the number of test inputs for which the neural
        network outputs the correct result."""

        # Initialize test_results if it's not provided
        if test_results is None:
            test_results = []

        # Collect predictions and true labels
        for (x, y) in test_data:
            arr = self.feedforward(x)
            test_results.append((arr, y))

        y_true = [true for _, true in test_results]
        #print("True labels:", y_true)

        # Assuming the network outputs an array, we can use argmax to find the most activated neuron
        y_pred = [np.argmax(pred) for pred, _ in test_results]
        y_true = [np.argmax(true) for true in y_true]

        #print("Predicted labels:", y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
