import random
import numpy as np
from typing import Optional, List
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction

class MultilayerPerceptron(object):
    
    def __init__(self, seed: Optional[int], topology , optimizer: Optimizer, weights=None, biases=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_layers = len(topology)
        self.optimizer= optimizer
        self.topology = topology

        self.sizes = [layer.num_neurons for layer in topology]
        self.activations = [layer.activation_function for layer in topology]

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]  
            self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w, activation_func in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, a) + b
            if activation_func:
                a = activation_func.activation(z)
            else:
                a = z 
        return a

    def parallel_update(self, mini_batch):
        nabla_b, nabla_w = self.backprop(mini_batch)
        return nabla_b, nabla_w
    
    def fit(self, training_data, epochs, mini_batch_size, test_data=None):
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print( f"Epoch {j} complete")

            

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

   
    def update_mini_batch(self, mini_batch):
        
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
        for b, w, activation_func in zip(self.biases, self.weights, self.activations):
            z = np.dot( w ,activation) +b
            zs.append(z)
            if activation_func:
                activation = activation_func.activation(z)
            else:
                activation = z  
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y)
        if self.activations[-1]:
            delta *= self.activations[-1].activation_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] =  np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            activation_func = self.activations[-l]
            if activation_func:
                sp = activation_func.activation_prime(z)
                delta = np.dot(self.weights[-l+1].T, delta) * sp
            else:
                delta = np.dot(self.weights[-l+1].T, delta)
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
        topology_array = np.array(self.topology, dtype=object)

        np.savez_compressed(filename, weights=weights_array, biases=biases_array, topology=topology_array)


    @classmethod
    def load_model(cls, filename, optimizer, seed=None):
        data = np.load(filename, allow_pickle=True)
        weights = data['weights'].tolist()
        biases = data['biases'].tolist()
        topology = data['topology'].tolist()

        model = cls(
            seed=seed,
            topology=topology,
            optimizer=optimizer,
            weights=weights,
            biases=biases
        )
        return model
