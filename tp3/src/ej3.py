from models.mlp.network import MultilayerPerceptron
from utils.activation_function import ActivationFunction
from utils.optimizer import Optimizer
import numpy as np
import os
import json
import sys
from typing import NamedTuple, Optional

RESULTS_DIR="output/ej3"
XOR_FILE = "xor.json"
PARITY_FILE = "parity.json"
DIGIT_FILE = "digit.json"

class Config(NamedTuple):
    type: Optional[str] = None
    data: Optional[str] = None
    output: Optional[str] = None
    topology: Optional[list] = None
    activation_function: Optional[ActivationFunction] = None
    optimizer: Optional[Optimizer] = None
    epochs: Optional[int] = None
    mini_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    seed: Optional[int] = None



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
    
    net = MultilayerPerceptron(
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

    training_data = list(zip(x, y))

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
        eta=config.learning_rate,
    ) # ! learning rate is divided by the mini_batch_update

    print(f"Accuracy: {net.evaluate(training_data)}")
    return 1




################################################################################################################################################

def persist_results(json_file: str, weights: list[np.ndarray], biases: list[np.ndarray], test_results: list[list[tuple[np.ndarray, np.ndarray]]], epochs: int) -> None:
    # Create directory if it does not exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

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


################################################################################################################################################



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as file:
        data = json.load(file)

  

    config = Config(
        type=data['problem'].get("type", None),  # Uses None if 'type' is missing
        data=data['problem'].get("data", None),
        output=data['problem'].get("output", None),
        topology=data['network'].get('topology', None),
        activation_function=ActivationFunction(
            method=data['network']['activation_function'].get('method', None),
            beta=data['network']['activation_function'].get('beta', None)
        ),
        optimizer=Optimizer(
            method=data['network']['optimizer'].get('method', None),
            eta=data['training'].get('learning_rate', None), # nasty sori
            beta1=data['network']['optimizer'].get('beta1', None),
            beta2=data['network']['optimizer'].get('beta2', None),
            epsilon=data['network']['optimizer'].get('epsilon', None),
            m=data['network']['optimizer'].get('m', None),
            v=data['network']['optimizer'].get('v', None),
            t=data['network']['optimizer'].get('t', None)
        ),
        epochs=data['training'].get('epochs', None),
        mini_batch_size=data['training'].get('mini_batch_size', None),
        learning_rate=data['training'].get('learning_rate', None),
        epsilon=data['training'].get('epsilon', None),
        seed=data['training'].get('seed', None)
    )



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
