import random
import os
import numpy as np
from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):

    def __init__(self, seed, topology, activation_function, optimizer):
        if seed is not None:
            np.random.seed(seed)
        self.num_layers = len(topology)
        self.topology = topology
        self.biases = [np.random.randn(y, 1) for y in topology[1:]] # crea vector de bias para cada capa
        self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])] # crea matriz de pesos para cada lazo
        self.activation_function = activation_function 
        self.optimizer = optimizer

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.activation(np.dot(w, a) + b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta, epsilon, test_data=None):
        if test_data is not None: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data=test_data, epsilon=epsilon), n_test))
            else:
                print("Epoch {0} complete".format(j))
            

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
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


    def backprop(self, predicted, expected):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = predicted
        activations = [predicted]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)
        
        
        delta = self.cost_derivative(activations[-1], expected) * self.activation_function.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function.activation_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, epsilon, test_results=None):
        for (x, y) in test_data:
            arr = self.feedforward(x)
            if test_results is not None:
                test_results.append((arr, y))
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        a = sum(
            int(np.all(np.abs(x - y) < epsilon))
            for (x, y) in test_results
        )
        return a

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
