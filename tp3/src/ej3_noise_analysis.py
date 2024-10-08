from models.mlp.network_2 import MultilayerPerceptron as MLP2
from models.mlp.network import MultilayerPerceptron
from utils.activation_function import ActivationFunction
from utils.optimizer import Optimizer
import numpy as np
import os
import json
import sys
from utils.config import Config

RESULTS_DIR="output/ej3"
XOR_FILE = "xor.json"
PARITY_FILE = "parity.json"
CLEAN_CLEAN_DIR = "clean_clean"
CLEAN_NOISY1_DIR = "clean_noisy1"
NOISY1_NOISY1_DIR = "noisy1_noisy1"
NOISY1_NOISY2_DIR = "noisy1_noisy2"
DIGIT_TRAIN_FILE = "digit_train.json"
DIGIT_TEST_FILE = "res_digit.json"
SECOND_TRAIN_SALT_PEPPER = "second_train_salt_pepper"

def convert_file_to_numpy(file_path: str, bits_per_element: int) -> np.ndarray:
    with open(file_path, 'r') as file:
        data = file.read().splitlines()

    output = []
    current_element = []

    for line in data:
        row = [int(char) for char in line.replace(" ", "")]

        current_element.extend(row)

        while len(current_element) >= bits_per_element:
            element_bits = current_element[:bits_per_element]
            output.append([[bit] for bit in element_bits])
            current_element = current_element[bits_per_element:]

    if current_element:
        raise ValueError(f"Leftover bits that do not form a complete element: {current_element}")

    return np.array(output)

# Exercise 1
def logic_xor(config: Config):
    X_logical = convert_file_to_numpy(config.data, bits_per_element=2)
    y_selected = np.array([[0], [1], [1], [0]])
    training_data = list(zip(X_logical, y_selected))

    net = MLP2(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    test_data = list(zip(X_logical, y_selected))

    test_results: list[tuple[int, int]] = []

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
        eta=config.learning_rate,
        epsilon=config.epsilon,
        test_data=test_data,
        test_results=test_results
    )

    persist_results(f'{RESULTS_DIR}/{XOR_FILE}', net.weights, net.biases, test_results, config.epochs)

    print(f"Accuracy: {net.evaluate(test_data=training_data, epsilon=config.epsilon)}")






# Exercise 2
def parity(config: Config):
    x = convert_file_to_numpy(config.data, bits_per_element=35)
    # Output data (even = 0, odd = 1)
    y = np.array([
        [0],  [1],  [0],  [1],  [0],  [1],  [0],  [1],  [0],  [1],
    ])

    training_data = list(zip(x, y))

    # Our logic which the net probably doesnt follow is taking the 35 bits as input
    # Second layer hopefully identifies likelyhood of being each number (0, 1, 2...)
    # Last layer simply activates taking into account only the neurons that represent the even numbers

    net = MLP2(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    test_results: list[tuple[int, int]] = []

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
        eta=config.learning_rate,
        epsilon=config.epsilon,
        test_data=training_data,
        test_results=test_results
        ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{PARITY_FILE}', net.weights, net.biases, test_results, config.epochs)

    print(f"Accuracy: {net.evaluate(test_data=training_data, epsilon=config.epsilon)}")
    return 1

# Exercise 3
def number_identifier(config: Config):
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    while len(y) < len(x):
        y = np.append(y, y, axis=0)

    training_data = list(zip(x, y))

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    test_results: list[tuple[int, int]] = []

    test_data = None
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if config.testing_data is not None:
        test_data = list(zip(
            convert_file_to_numpy(config.testing_data, bits_per_element=35),
            [np.array([[1] if i == digit else [0] for digit in digits]) for i in range(10)]
        ))

    net.fit_with_cross_validation(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        n_splits=config.n_splits,
        test_results=test_results
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)

    test_results = []

    print(f"Accuracy: {net.evaluate(test_data, epsilon=config.epsilon, test_results=test_results)}")

    # wrap each element in a list
    test_results = [test_results]

    persist_results(f'{RESULTS_DIR}/{DIGIT_TRAIN_FILE}', net.weights, net.biases, test_results, config.epochs)

    return 1

##########################################################################################################
##########################################################################################################
##########################################################################################################

def number_identifier_clean_clean(config: Config):
    """ Trains a network with clean digits and test it with clean digits.
       Then saves the results of the test in a json file """
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    training_data = list(zip(x, y))

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer  
    )

    test_results: list[tuple[int, int]] = []

    test_data = None
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if config.testing_data is not None:
        test_data = list(zip(
            convert_file_to_numpy(config.testing_data, bits_per_element=35),
            [np.array([[1] if i == digit else [0] for digit in digits]) for i in range(10)]
        ))

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        test_data=test_data,
        test_results=test_results
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{CLEAN_CLEAN_DIR}/{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)

    return 1

def number_identifier_clean_noisy1(config: Config):
    """ Trains a network with clean digits and test it with noisy digits """
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    training_data = list(zip(x, y))

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer  
    )

    test_results: list[tuple[int, int]] = []
    training_results: list[tuple[int, int]] = []

    test_data = None
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if config.testing_data is not None:
        test_data = list(zip(
            convert_file_to_numpy(config.testing_data, bits_per_element=35),
            [np.array([[1] if i == digit else [0] for digit in digits]) for i in range(10)]
        ))

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        test_data=test_data,
        test_results=test_results,
        training_results=training_results
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{CLEAN_NOISY1_DIR}/{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)
    persist_results(f'{RESULTS_DIR}/{CLEAN_NOISY1_DIR}/{DIGIT_TRAIN_FILE}', net.weights, net.biases, training_results, config.epochs)

    return 1

def number_identifier_noisy1_noisy1(config: Config):
    """ Trains a network with noisy digits and test it with noisy digits with the same distribution.
      The weights and biases that the network uses are taken from the clean_clean training.
       It then saves the results of the test in a json file """
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    while len(y) < len(x):
        y = np.append(y, y, axis=0)

    training_data = list(zip(x, y))

    weights, biases = parse_weights_and_biases(config.path_to_weights_and_biases)

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer,
        weights=weights,
        biases=biases
    )

    test_results: list[tuple[int, int]] = []
    training_results: list[tuple[int, int]] = []

    test_data = None
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if config.testing_data is not None:
        test_data = list(zip(
            convert_file_to_numpy(config.testing_data, bits_per_element=35),
            [np.array([[1] if i == digit else [0] for digit in digits]) for i in range(10)]
        ))

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        test_data=test_data,
        test_results=test_results,
        training_results=training_results
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{NOISY1_NOISY1_DIR}/{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)

    persist_results(f'{RESULTS_DIR}/{NOISY1_NOISY1_DIR}/{DIGIT_TRAIN_FILE}', net.weights, net.biases, training_results, config.epochs)

    return 1

def number_identifier_noisy1_noisy2(config: Config):
    """ Trains a network with noisy digits and test it with noisy digits with another distribution.
      The weights and biases that the network uses are taken from the clean_clean training.
       It then saves the results of the test in a json file """
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    while len(y) < len(x):
        y = np.append(y, y, axis=0)

    training_data = list(zip(x, y))

    weights, biases = parse_weights_and_biases(config.path_to_weights_and_biases)

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer,
        weights=weights,
        biases=biases
    )

    test_results: list[tuple[int, int]] = []

    net.fit_with_cross_validation(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        n_splits=config.n_splits,
        test_results=test_results,
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{NOISY1_NOISY2_DIR}/cross_val_{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)

    return 1

def number_identifier_second_train_salt_pepper(config: Config):
    """ Trains a network with noisy digits and test it with noisy digits with another distribution.
      The weights and biases that the network uses are taken from the clean_clean training.
       It then saves the results of the test in a json file """
    x = convert_file_to_numpy(config.data, bits_per_element=35)

    y = np.array([
        [[1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1],  [0]],
        [[0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [1]],
    ])

    while len(y) < len(x):
        y = np.append(y, y, axis=0)

    training_data = list(zip(x, y))

    weights, biases = parse_weights_and_biases(config.path_to_weights_and_biases)

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer,
        weights=weights,
        biases=biases
    )

    test_results: list[tuple[int, int]] = []

    net.fit_with_cross_validation(
        training_data=training_data,
        epochs=config.epochs,
        eta=config.learning_rate,
        mini_batch_size=config.mini_batch_size,
        epsilon=config.epsilon,
        n_splits=config.n_splits,
        test_results=test_results,
    ) # ! learning rate is divided by the mini_batch_update

    persist_results(f'{RESULTS_DIR}/{SECOND_TRAIN_SALT_PEPPER}/cross_val_{DIGIT_TEST_FILE}', net.weights, net.biases, test_results, config.epochs)

    return 1

################################################################################################################################################

def persist_results(json_file: str, weights: list[np.ndarray], biases: list[np.ndarray], test_results: list[list[tuple[np.ndarray, np.ndarray]]], epochs: int) -> None:
    # Create directory if it does not exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))

    def convert_test_results(test_results):
        # Convert numpy arrays to lists and unwrap single-element arrays
        def unwrap(array):
            if array.size == 1:
                return array.item()
            else:
                return array.tolist()
        
        return [[(unwrap(pred), unwrap(true)) for pred, true in epoch] for epoch in test_results]

    data = {
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases],
        "test_results": convert_test_results(test_results),
        "epochs": epochs
    }
    
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def parse_weights_and_biases(file_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if file_path is None or not os.path.exists(file_path):
        return None, None
    with open(file_path, 'r') as file:
        data = json.load(file)

    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]

    return weights, biases

################################################################################################################################################



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])

    # Now you can use these NamedTuple instances in your logic
    if config.type == "xor":
        logic_xor(config)
    elif config.type == "parity":
        parity(config)
    elif config.type == "number_identifier":
        number_identifier(config)
    elif config.type == "number_identifier_1":
        number_identifier_clean_clean(config)
    elif config.type == "number_identifier_2":
        number_identifier_clean_noisy1(config)
    elif config.type == "number_identifier_4":
        number_identifier_noisy1_noisy2(config)
    elif config.type == "number_identifier_5":
        number_identifier_second_train_salt_pepper(config)
    else:
        print("Invalid problem type")
        sys.exit(1)




if __name__ == "__main__":
    main()
