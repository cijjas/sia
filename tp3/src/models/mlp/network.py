"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import os



# Third-party libraries
import numpy as np
from typing import Optional
import json


from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):

    def __init__(self, seed, topology:list[int], activation_function: ActivationFunction, optimizer:Optimizer):
        """The list ``topology`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        if seed is not None:
            np.random.seed(seed)
        self.num_layers: int = len(topology)
        self.topology: list[int] = topology
        self.biases: list[np.ndarray] = [np.random.randn(y, 1) for y in topology[1:]] # crea vector de bias para cada capa
        self.weights: list[np.ndarray] = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])] # crea matriz de pesos para cada lazo
        self.activation_function = activation_function
        self.optimizer = optimizer

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if 'a' is the input."""
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.activation(np.dot(w, a) + b)
        return a

    def fit(self, training_data: list[tuple[np.ndarray, np.ndarray]], epochs: int, mini_batch_size: int, eta: float, epsilon: float, test_data: Optional[list[tuple[np.ndarray, int]]] = None) -> None:
        # Optional is from python 2.7, it is used to indicate that a parameter is optional
        # Here in python 3 we can use the optional this way: Optional[type]
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data is not None: n_test = len(test_data)
        n: int = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches:list[list[tuple[np.ndarray, np.ndarray]]] = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data=test_data, epsilon=epsilon), n_test))
            else:
                print("Epoch {0} complete".format(j))


    def update_mini_batch(self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float) -> None:
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


    def backprop(self, predicted:np.ndarray, expected:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b: list[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        nabla_w: list[np.ndarray] = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation: np.ndarray = predicted
        # activations is a list of arrays, where each array is the activations of the neurons in that layer
        activations: list[np.ndarray] = [predicted] # list to store all the activations, layer by layer
        zs: list[np.ndarray] = [] # list to store all the z vectors, layer by layer
        # Feed Foward that stores the values of the activations for later usage
        for b, w in zip(self.biases, self.weights):
            z: np.ndarray = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)
        # Backpropagation
        # Last layer needs special treatment
        delta: np.ndarray = self.cost_derivative(activations[-1], expected) * self.activation_function.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        # Now we go from the second to last to the input layer
        for l in range(2, self.num_layers):
            z: np.ndarray = zs[-l]
            sp: np.ndarray = self.activation_function.activation_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: list[tuple[np.ndarray, int]], epsilon: float, test_results: list[tuple[int, int]] = None) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        for (x, y) in test_data:
            arr: np.ndarray = self.feedforward(x)
            if test_results is not None:
                test_results.append((arr, y))

        # test_results: list[tuple[int, int]] = [(np.argmax(self.feedforward(x)), y)
        #                for (x, y) in test_data]
        test_results: list[tuple[int, int]] = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        a = sum(
            int(np.all(np.abs(x - y) < epsilon))
            for (x, y) in test_results
        )
        return a

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the vector of partial derivatives partial C_x
        partial a for the output activations."""
        return (output_activations-y)
