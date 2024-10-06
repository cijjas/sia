import random
import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score

from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):
    
    def __init__(self, seed,  sizes, activation_function: ActivationFunction, optimizer: Optimizer, weights=None, biases=None):
        if seed is not None:
            np.random.seed(seed)
      
        self.num_layers = len(sizes)
        self.sigma = activation_function
        self.optimizer= optimizer
        self.sizes = sizes
        self.history = {"loss": [], "accuracy": [], "grad_magnitudes": []}

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigma.activation(np.dot(w, a)+b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                self.history["accuracy"].append(self.evaluate(test_data))
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print( f"Epoch {j} complete")

            loss = self.calculate_loss(training_data)
            grad_magnitude = self.calculate_gradient_magnitude()
            
            self.history["loss"].append(loss)
            self.history["grad_magnitudes"].append(grad_magnitude)

    def calculate_loss(self, training_data):
        
        total_loss = 0
        for x, y in training_data:
            output = self.feedforward(x)
            total_loss += np.sum(self.cost_derivative(output, y) ** 2)
        return total_loss / len(training_data)

    def calculate_gradient_magnitude(self):
        total_grad_w = sum(np.linalg.norm(w) for w in self.weights)
        total_grad_b = sum(np.linalg.norm(b) for b in self.biases)
        return total_grad_w + total_grad_b

    def gradient_descent(self, weights, biases, grads_w, grads_b, mini_batch_size):
        weights = [w - (self.eta / mini_batch_size) * gw for w, gw in zip(weights, grads_w)]
        biases = [b - (self.eta / mini_batch_size) * gb for b, gb in zip(biases, grads_b)]
        return weights, biases

    def update_mini_batch(self, mini_batch, eta):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                
        self.weights, self.biases = self.optimizer.update_parameters(self.weights, self.biases, nabla_w, nabla_b)

    def backprop(self, x, y):
       
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot( w ,activation) +b
            zs.append(z)
            activation = self.sigma.activation(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigma.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] =  np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp =  self.sigma.activation_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
       
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
       
        return (output_activations-y)
    
    def save_model(self, filename):
        
        weights_array = np.array(self.weights, dtype=object)
        biases_array = np.array(self.biases, dtype=object)
        sizes_array = np.array(self.sizes, dtype=object)

        np.savez_compressed(filename, weights=weights_array, biases=biases_array, sizes=sizes_array)


    @classmethod
    def load_model(cls, filename, activation_function, optimizer, seed=None):
        data = np.load(filename, allow_pickle=True)
        weights = data['weights']
        biases = data['biases']
        sizes = data['sizes'].tolist()

        weights = weights.tolist()
        biases = biases.tolist()

        model = cls(
            seed=seed,
            sizes=sizes,
            activation_function=activation_function,
            optimizer=optimizer,
            weights=weights,
            biases=biases
        )
        return model
