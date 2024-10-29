import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.radius_function import str_to_radius_function
from utils.eta_function import str_to_eta_function
from utils.similarity_function import str_to_similarity_function
from sklearn.cluster import KMeans

class Kohonen:
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
        init_method="random_choice"
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
        elif init_method == "random_choice":
            self.weights = data[
                np.random.choice(
                    range(data.shape[0]), size=(grid_size, grid_size)
                )
            ]

        self.bmu_count = np.zeros((grid_size, grid_size))
        self.bmu_count_history = []
        self.bmu_mapping = {}

        # New lists to store errors
        self.quantization_errors = []
        self.topographic_errors = []

        # Image store for the GIFFF
        self.weight_snapshots = []

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
        neighborhood = self.get_neighborhood(bmu, iteration, max_iterations)
        lr = self.eta_function(self.learning_rate, iteration, max_iterations)

        neighborhood = neighborhood[..., np.newaxis]
        np.add(
            self.weights, lr * neighborhood * (sample - self.weights), out=self.weights
        )

    def fit(self, num_iterations):
        num_samples = self.input_data.shape[0]

        for epoch in range(num_iterations // num_samples):
            shuffled_indices = np.random.permutation(num_samples)

            for iteration in range(num_samples):
                sample = self.input_data[shuffled_indices[iteration]]
                bmu = self.find_bmu(sample)

                if bmu not in self.bmu_mapping:
                    self.bmu_mapping[bmu] = []
                self.bmu_mapping[bmu].append(shuffled_indices[iteration])

                self.update_weights(
                    sample, bmu, epoch * num_samples + iteration, num_iterations
                )

                ######## Varibles used for graphs and further analysis
                self.bmu_count[bmu] += 1
                self.bmu_count_history.append(self.bmu_count.copy())


                self.quantization_errors.append(self.calculate_quantization_error())
                self.topographic_errors.append(self.calculate_topology_error())

            self.weight_snapshots.append(self.weights.copy())

    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

    # This error takes the BMU for a specific input data element and checks how much the node and the data differ, higher difference, higher value
    def calculate_quantization_error(self):
        """ Calculate the quantization error """
        errors = []
        for sample in self.input_data:
            bmu = self.find_bmu(sample)
            error = np.linalg.norm(sample - self.weights[bmu])
            errors.append(error)
        return np.mean(errors)

    # This error checks than the BMU and the Second BMU are neighbors node-wise, this is a binary function, 1 if they are neighbors, 0 if they are not
    # So a high topological error means that the BMU and Second BMU arent usually neighbors, which means the topology of the original data is not maintained
    def calculate_topology_error(self):
        """ Calculate the topographic error. """
        topographic_error_count = 0
        for sample in self.input_data:
            bmu = self.find_bmu(sample)

            # Find the second-best matching unit (SBMU) by excluding the BMU
            distances = self.similarity_function(self.weights, sample)
            distances[bmu] = np.inf  # Exclude BMU by setting distance to infinity
            sbmu = np.unravel_index(np.argmin(distances, axis=None), self.bmu_count.shape)

            # Check if SBMU is adjacent to BMU
            if abs(bmu[0] - sbmu[0]) > 1 or abs(bmu[1] - sbmu[1]) > 1:
                topographic_error_count += 1

        return topographic_error_count / len(self.input_data)

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

    # Used for distance average graph
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

    # Used for variable heatmaps
    def calculate_variable_matrix(self, feature_column):
        variable_matrix = np.zeros((self.grid_size, self.grid_size))

        for bmu, indices in self.bmu_mapping.items():
            feature_values = self.input_data[indices, feature_column]
            variable_matrix[bmu] = np.mean(feature_values)

        return variable_matrix
