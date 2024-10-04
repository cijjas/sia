import numpy as np
from utils.optimizer import Optimizer
from utils.activation_function import ActivationFunction
from numba import njit

class MultilayerPerceptron:


    def __init__(self, seed, topology, activation_function, optimizer, weights=None, biases=None):
        
        if seed is not None:
            np.random.seed(seed)
        self.num_layers = len(topology)
        self.topology = topology
        self.activation_function = activation_function 
        self.optimizer = optimizer

        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.biases = [np.random.randn(y, 1) for y in topology[1:]]  
            self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])]

    @njit
    def feedforward(self, inputs):
        
        activations = inputs
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activations) + bias
            activations = self.activation_function.activation(z)
        return activations

    def fit(self, training_data, epochs, mini_batch_size, eta, epsilon, test_data=None):
       
        n = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                evaluation = self.evaluate(test_data, epsilon)
                print(f"Epoch {epoch}: {evaluation} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    @njit
    def update_mini_batch(self, mini_batch, eta):
    # Stack inputs and outputs into matrices
        x_batch = np.hstack([x for x, _ in mini_batch])
        y_batch = np.hstack([y for _, y in mini_batch])

        # Perform backpropagation
        nabla_b, nabla_w = self.backpropagation(x_batch, y_batch)

        # Update weights and biases
         # Update weights and biases
        self.weights, self.biases = self.optimizer.update(
            weights=self.weights,
            biases=self.biases,
            grads_w=nabla_w,
            grads_b=nabla_b,
            mini_batch_size=len(mini_batch)  # Use the actual mini-batch size
        )

    @njit
    def backpropagation(self, x_batch, y_batch):
       
        batch_size = x_batch.shape[1]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activation = x_batch
        activations = [x_batch]
        zs = []

        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            zs.append(z)
            activation = self.activation_function.activation(z)
            activations.append(activation)

        # Backward pass
        delta = (self.cost_derivative(activations[-1], y_batch) *
                 self.activation_function.activation_prime(zs[-1]))
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function.activation_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        # Average gradients over the batch
        nabla_b = [nb / batch_size for nb in nabla_b]
        nabla_w = [nw / batch_size for nw in nabla_w]

        return nabla_b, nabla_w

    @njit
    def evaluate(self, test_data, epsilon):
        
        x_test = np.hstack([x for x, _ in test_data])
        y_test = np.hstack([y for _, y in test_data])

        output = self.feedforward(x_test)
        correct_predictions = np.sum(np.all(np.abs(output - y_test) < epsilon, axis=0))
        return correct_predictions

    @njit
    def cost_derivative(self, output_activations, y_batch):
        
        return output_activations - y_batch

    def save_model(self, filename):
        
        weights_array = np.array(self.weights, dtype=object)
        biases_array = np.array(self.biases, dtype=object)
        topology_array = np.array(self.topology, dtype=object)

        # Save weights, biases, and topology to a npz file
        np.savez_compressed(filename, weights=weights_array, biases=biases_array, topology=topology_array)


    @classmethod
    def load_model(cls, filename, activation_function, optimizer, seed=None):
       # Load weights and biases from a npz file
        data = np.load(filename, allow_pickle=True)
        weights = data['weights']
        biases = data['biases']
        topology = data['topology'].tolist()

        # Convert weights and biases back to lists
        weights = weights.tolist()
        biases = biases.tolist()

        model = cls(
            seed=seed,
            topology=topology,
            activation_function=activation_function,
            optimizer=optimizer,
            weights=weights,
            biases=biases
        )
        return model