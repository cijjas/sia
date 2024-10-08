import numpy as np
from enum import Enum

NUMBERS = 10

class NoiseType(Enum):
    GAUSSIAN = 1
    SALT_AND_PEPPER = 2

class NoiseFactory:
    
    @staticmethod
    def generate_zeros(size=(7, 5)) -> np.ndarray:
        return np.zeros(size, dtype=int)
    
    @staticmethod
    def add_gaussian_noise(binary_matrix, noise_mean=0, noise_std=0.5):
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=binary_matrix.shape)
        noisy_matrix = binary_matrix + noise
        noisy_matrix = np.where(noisy_matrix >= 0.5, 1, 0)
        return noisy_matrix

    @staticmethod
    def add_salt_and_pepper_noise(binary_matrix, salt_prob=0.05, pepper_prob=0.05):
        noisy_matrix = np.copy(binary_matrix)
        for row in range(binary_matrix.shape[0]):
            for col in range(binary_matrix.shape[1]):
                rand = np.random.random()
                if rand < salt_prob:
                    noisy_matrix[row, col] = 1
                elif rand > 1 - pepper_prob:
                    noisy_matrix[row, col] = 0
        return noisy_matrix
    
    @staticmethod
    def save_to_file(matrix, file_path):
        np.savetxt(file_path, matrix, fmt='%d')

    @classmethod
    def generate_noisy_numbers(cls, noise_type: NoiseType, file_path: str, config=None):
        binary_matrix = cls.generate_zeros()
        
        if noise_type == NoiseType.GAUSSIAN:
            noisy_matrix = cls.add_gaussian_noise(binary_matrix, noise_mean=config.get('mean', 0), noise_std=config.get('stddev', 0.5))
            numbers = int(config.get('numbers', NUMBERS))
            for _ in range(numbers-1):
                noisy_matrix = np.append(noisy_matrix, cls.add_gaussian_noise(binary_matrix, noise_mean=config.get('mean', 0), noise_std=config.get('stddev', 0.5)), axis=0)
        elif noise_type == NoiseType.SALT_AND_PEPPER:
            noisy_matrix = cls.add_salt_and_pepper_noise(binary_matrix, salt_prob=config.get('salt_prob', 0.05), pepper_prob=config.get('pepper_prob', 0.05))
            numbers = int(config.get('numbers', NUMBERS))
            for _ in range(numbers-1):
                noisy_matrix = np.append(noisy_matrix, cls.add_salt_and_pepper_noise(binary_matrix, salt_prob=config.get('salt_prob', 0.05), pepper_prob=config.get('pepper_prob', 0.05)), axis=0)
        else:
            raise ValueError('Invalid noise type')
        
        cls.save_to_file(noisy_matrix, file_path)


