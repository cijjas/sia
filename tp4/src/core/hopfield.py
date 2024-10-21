import numpy as np


# Associative memory model based on Hopfield networks
# https://en.wikipedia.org/wiki/Hopfield_network
# https://www.youtube.com/watch?v=1WPJdAW-sFo
class Hopfield:
    def __init__(self, n_neurons):

        self.n_neurons = n_neurons
        # As every neuron is connected to every other neuron, the weight matrix is fully connected
        # Excluding self-connections (diagonal), the weight matrix is symmetric
        self.W = np.zeros((n_neurons, n_neurons))
        self.state_history = []

    # patterns (ndarray): Array of shape (n_patterns, n_neurons), with elements -1 or 1.
    def train(self, patterns):
        # Train the network using Hebbian learning by updating the weight matrix
        # based on the outer product of each pattern with itself
        # this step is creating the low-energy states that the network will converge to
        np.sum([np.outer(pattern, pattern) for pattern in patterns], axis=0, out=self.W)
        # Ensure zero diagonal (no self-connections)
        np.fill_diagonal(self.W, 0)
        # Normalize the weights

    def update(self, input_pattern, max_steps=5, is_synchronous=False):
        current_state = input_pattern.copy()
        
        for _ in range(max_steps):
            if is_synchronous:
                # Synchronously update the entire state at once
                current_state = np.sign(self.W @ current_state)
                current_state[current_state == 0] = 1  # Replace any zeros with 1
            else:
                # Asynchronously update each neuron based on its net input
                for neuron_index in range(self.n_neurons):
                    # Calculate the net input for the current neuron
                    net_input = np.dot(self.W[neuron_index], current_state)
                    # Update the state of the neuron based on the sign of the net input
                    current_state[neuron_index] = 1 if net_input >= 0 else -1
                    self.state_history.append(current_state.copy())
                    
        return current_state, self.state_history


   

    def energy(self, pattern):
        # The network is governed by an energy function
        # that determines the stability of the stored patterns.
        # Memories are low-energy states, and the network tends to converge to them.
        # E(p) = -0.5 * p^T W p
        return -0.5 * pattern @ self.W @ pattern
