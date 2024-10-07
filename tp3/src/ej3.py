from models.mlp.network_2 import MultilayerPerceptron
import numpy as np
import sys
from utils.config import Config

RESULTS_DIR="output/ej3"
XOR_FILE = "xor.json"
PARITY_FILE = "parity.json"
DIGIT_FILE = "digit.json"


# Exercise 1
def logic_xor(config: Config):
    X_logical = convert_file_to_numpy(config.data, bits_per_element=2)
    y_selected = np.array([[0], [1], [1], [0]])
    training_data = list(zip(X_logical, y_selected))
    
    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology, 
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    net.fit(
        training_data=training_data,
        epochs=config.epochs, 
        mini_batch_size=config.mini_batch_size,
    )

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


    net = MultilayerPerceptron(
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
       
        ) 
    

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

    training_data = list(zip(x, y))

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer,
        weights=weights,
        biases=biases
    )
    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
    ) 

    test_results: list[tuple[int, int]] = []

    test_data = None
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    if config.testing_data is not None:
        test_data = list(zip(
            convert_file_to_numpy(config.testing_data, bits_per_element=35),
            [np.array([[1] if i == digit else [0] for digit in digits]) for i in range(10)]
        ))

    


    print(f"Accuracy: {net.evaluate(test_data, epsilon=config.epsilon)}")
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


############################################################################################################

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])

    # Now you can use these NamedTuple instances in your logic
    if config.type == "xor":
        logic_xor(config)
    elif config.type == "number_identifier":
        number_identifier(config)
    elif config.type == "parity":
        parity(config)
    else:
        print("Invalid problem type")
        sys.exit(1)

if __name__ == "__main__":
    main()