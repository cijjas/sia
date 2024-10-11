import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.radius_function import constant_radius
from utils.eta_function import constant_eta
from utils.similarity_function import euclidean_distance

class Kohonen:

    # grid_size = k
    def __init__(self, data, grid_size, learning_rate=0.5, eta_function:callable=constant_eta, radius=1, 
                 radius_function=constant_radius, similarity_function=euclidean_distance, seed=None, weights:np.ndarray=None):
        if seed is not None:
            np.random.seed(seed)
        self.input_data = data
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.eta_function = eta_function
        self.radius = radius
        self.radius_function = radius_function
        self.similarity_function = similarity_function

        if weights is not None:
            self.weights = weights
        else:
            self.weights: np.ndarray = data[
                np.random.choice(
                    range(data.shape[0]), size=(grid_size, grid_size)
                )  # initialize a k*k matrix with random samples from training data
            ]

        self.bmu_count = np.zeros((grid_size, grid_size))
        self.bmu_count_history = []
        self.bmu_mapping = {}  # This will store data point indices mapped to each BMU

    def find_bmu(self, sample):
        distances = self.similarity_function(self.weights, sample, axis=-1)
        return np.unravel_index(np.argmin(distances, axis=None), self.bmu_count.shape)

    def get_neighborhood(self, bmu, iteration, max_iterations):
        radius = self.radius_function(self.radius, iteration, max_iterations)
        distances = np.linalg.norm(
            np.indices((self.grid_size, self.grid_size)).T - np.array(bmu), axis=-1
        )  # distance of each neuron to the BMU
        return np.exp(-(distances**2) / (2 * (radius**2)))  # gaussian function

    def update_weights(self, sample, bmu, iteration, max_iterations):
        neighborhood = self.get_neighborhood(bmu, iteration, max_iterations)
        lr = self.eta_function(self.learning_rate, iteration, max_iterations)

        # add dimension to neighborhood
        # th shape of neighborhood is (k,k) and the shape of weights is (k,k,features)
        # now the shape of neighborhood is (k,k,1)
        # sample has shape (features,)
        # sample - self.weights has shape (k,k,features)
        neighborhood = neighborhood[..., np.newaxis]
        self.weights += lr * neighborhood * (sample - self.weights)

    def train(self, num_iterations):
        for iteration in range(num_iterations):
            # select a random sample from the training data
            sample_index = np.random.randint(self.input_data.shape[0])
            sample = self.input_data[sample_index]
            # find the winning neuron such that the weights of the neuron are closest to the sample
            # BMU can be defined in many ways, here we use the Euclidean distance
            bmu = self.find_bmu(sample)

            self.bmu_count[bmu] += 1
            self.bmu_count_history.append(self.bmu_count.copy())

            if bmu not in self.bmu_mapping:
                self.bmu_mapping[bmu] = []
            self.bmu_mapping[bmu].append(sample_index)

            # Update neighbouring neurons based on Kohonen learning rule
            self.update_weights(sample, bmu, iteration, num_iterations)

    def calculate_average_neighbor_distances(self):
        distances = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neuron_weight = self.weights[i, j]
                if i < self.grid_size - 1:
                    distances.append(
                        np.linalg.norm(neuron_weight - self.weights[i + 1, j])
                    )
                if j < self.grid_size - 1:
                    distances.append(
                        np.linalg.norm(neuron_weight - self.weights[i, j + 1])
                    )
        return np.mean(distances)

    def calculate_umatrix(self):
        u_matrix = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = []

                if i > 0:
                    neighbors.append(self.weights[i - 1, j])
                if i < self.grid_size - 1:
                    neighbors.append(self.weights[i + 1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j - 1])
                if j < self.grid_size - 1:
                    neighbors.append(self.weights[i, j + 1])

                if neighbors:
                    distances = [
                        np.linalg.norm(self.weights[i, j] - neighbor)
                        for neighbor in neighbors
                    ]
                    u_matrix[i, j] = np.mean(distances)

        return u_matrix

    def calculate_variable_matrix(self, feature_column):
        # Initialize matrix to store average values for each BMU
        variable_matrix = np.zeros((self.grid_size, self.grid_size))

        # Iterate over each BMU and calculate the average value for the mapped data points
        for bmu, indices in self.bmu_mapping.items():
            # Extract the values of the feature column for all points mapped to the BMU
            feature_values = self.input_data[indices, feature_column]
            # Calculate the average value for this BMU
            variable_matrix[bmu] = np.mean(feature_values)

        return variable_matrix
