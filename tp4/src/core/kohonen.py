import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.radius_function import str_to_radius_function
from utils.eta_function import str_to_eta_function
from utils.similarity_function import str_to_similarity_function


class Kohonen:

    # grid_size = k
    def __init__(
        self,
        data,
        grid_size,
        learning_rate=0.5,
        eta_function="constant",
        radius=1,
        radius_function="exponential_decay",
        similarity_function="euclidean",
        seed=None,
        weights: np.ndarray = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.input_data = data
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.radius = radius

        self.eta_function = str_to_eta_function(eta_function)
        self.radius_function = str_to_radius_function(radius_function)
        self.similarity_function = str_to_similarity_function(similarity_function)

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

    def find_bmu(self, sample) -> tuple[int, int]:
        distances = self.similarity_function(self.weights, sample)
        return np.unravel_index(np.argmin(distances, axis=None), self.bmu_count.shape)

    def get_neighborhood(self, bmu, iteration, max_iterations):
        radius = self.radius_function(self.radius, iteration, max_iterations)
        distances = np.linalg.norm(
            np.indices((self.grid_size, self.grid_size)).T - np.array(bmu), axis=-1
        )  # distance of each neuron to the BMU
        return np.exp(-(distances**2) / (2 * (radius**2)))  # gaussian function

    def update_weights(self, sample, bmu, iteration, max_iterations):
        # neighborhood is matrix of shape (k,k) with values between 0 and 1 representing a gaussian function
        neighborhood = self.get_neighborhood(bmu, iteration, max_iterations)
        lr = self.eta_function(self.learning_rate, iteration, max_iterations)

        # add dimension to neighborhood
        # th shape of neighborhood is (k,k) and the shape of weights is (k,k,features)
        # now the shape of neighborhood is (k,k,1)
        # sample has shape (features,)
        # sample - self.weights has shape (k,k,features)
        neighborhood = neighborhood[..., np.newaxis]
        np.add(
            self.weights, lr * neighborhood * (sample - self.weights), out=self.weights
        )

    def fit(self, num_iterations):
        num_samples = self.input_data.shape[0]
        for epoch in range(num_iterations // num_samples):
            # Shuffle the input data at the beginning of each epoch
            shuffled_indices = np.random.permutation(num_samples)

            for iteration in range(num_samples):
                sample = self.input_data[shuffled_indices[iteration]]
                bmu = self.find_bmu(sample)

                # --------- graph data --------------
                # Increment BMU count in place

                np.add(self.bmu_count[bmu], 1, out=self.bmu_count[bmu])
                self.bmu_count_history.append(self.bmu_count.copy())

                if bmu not in self.bmu_mapping:
                    self.bmu_mapping[bmu] = []
                self.bmu_mapping[bmu].append(shuffled_indices[iteration])
                # -----------------------------------

                self.update_weights(
                    sample, bmu, epoch * num_samples + iteration, num_iterations
                )

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
