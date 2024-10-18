import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PCA:

    def __init__(self, n_components=None, normalize=True):
        """
        Initializes PCA with the data and optional number of components.
        :param data: Data matrix (n_samples, n_features)
        :param n_components: Number of principal components to retain. If None, retain all.
        :param normalize: If True, normalize the data before applying PCA.
        """
        self.n_components = n_components
        self.normalize = normalize
        self.mean = None
        self.components = None
        self.expl_variance = None

    def normalize_data(self):

        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)

        np.subtract(self.data, self.mean, out=self.data)
        np.divide(self.data, self.std, out=self.data)

    def calculate_covariance_matrix(self):
        # Covariance: degree to which a pair of variables change together
        # s_ik = 1/n * sum_j (x_ij - mean_i) * (x_ij - mean_i)
        # s_ik > 0 -> positive association between the data of the variables
        # s_ik < 0 -> negative association between the data of the variables
        # s_ik = 0 -> no association between the data of the variables
        # s_ii = variance of variable i = cov(i,i) how much a the variable varies
        # rowvar = False -> each column represents a variable
        return np.cov(self.data, rowvar=False)

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        else:
            self.data = data

        if self.normalize:
            self.normalize_data()

        # Step 1: Calculate the covariance (featuresxfeatures) matrix
        # This matrixs represents the relationship between the variables
        covariance_matrix = self.calculate_covariance_matrix()

        # Step 2: Calculate eigenvalues and eigenvectors
        # eigenvectors: directions/axes of the data that maximize the variance
        # - AKA: directions in the feature space where the data is most spread out.
        # eigenvalues: magnitude of the variance in the direction of the eigenvector
        # - higher eigenvalue means that the data is spread out more in the direction of the corresponding eigenvector.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 3: Sort eigenvalues and eigenvectors
        # Sort the eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        # slect all the columns and sort them according to the sorted_idx
        eigenvectors = eigenvectors[:, sorted_idx]

        # Step 4: Retain only the top n_components eigenvectors
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, : self.n_components]
            eigenvalues = eigenvalues[: self.n_components]

        self.components = eigenvectors
        self.expl_variance = eigenvalues

    def transform(self, data):
        if self.components is None:
            raise ValueError("The model must be fitted before transforming the data.")

        data_transformed = np.copy(data)

        # Normalize the data using the stored mean and standard deviation
        if self.normalize:
            np.subtract(data_transformed, self.mean, out=data_transformed)
            np.divide(data_transformed, self.std, out=data_transformed)

        # Project the data onto the principal components
        return np.dot(data_transformed, self.components)

    def explained_variance_ratio(self):
        # Return the variance ratio only for the selected components with respect to the total variance
        total_variance = np.sum(self.data.var(axis=0))
        return self.expl_variance / total_variance

    def explained_variance(self):
        return self.expl_variance
