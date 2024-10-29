import numpy as np


# Associative memory model based on Hopfield networks
# https://en.wikipedia.org/wiki/Hopfield_network
# https://www.youtube.com/watch?v=1WPJdAW-sFo
class Hopfield:
    def __init__(self, n_neurons):
        np.random.seed(42)
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
        self.state_history = []
        self.energy_history = []

    def train(self, patterns):

        for pattern in patterns:
            self.W += np.outer(pattern, pattern)
        np.fill_diagonal(self.W, 0)
        self.W /= self.n_neurons

    def update(self, input_pattern, max_steps=5):
        current_state = input_pattern.copy()
        self.energy_history = []
        for _ in range(max_steps):
            neuron_indices = np.random.permutation(self.n_neurons)
            for neuron_index in neuron_indices:
                self.state_history.append(current_state.copy())
                self.energy_history.append(self.energy(current_state))
                net_input = np.dot(self.W[neuron_index], current_state)
                current_state[neuron_index] = 1 if net_input >= 0 else -1

        return current_state, self.state_history

    def energy(self, pattern):
        return -0.5 * np.dot(pattern, np.dot(self.W, pattern))
