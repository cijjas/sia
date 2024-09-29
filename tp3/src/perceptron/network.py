"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import os


# Third-party libraries
import numpy as np
from typing import Optional
import json
R_XOR_JSON = "xor.json"
RESULTS_DIR = "../results"

class MultilayerPerceptron(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers: int = len(sizes)
        self.sizes: list[int] = sizes
        self.biases: list[np.ndarray] = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights: list[np.ndarray] = [np.random.randn(y, x)
                                          for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a:np.ndarray) -> np.ndarray:
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            #print(f"b: {b}, w: {w}, a: {a}")
        return a

    def fit(self, training_data: list[tuple[np.ndarray, np.ndarray]], epochs: int, mini_batch_size: int, eta: float,
            test_data: Optional[list[tuple[np.ndarray, int]]] = None) -> None:
        # Optional is from python 2.7, it is used to indicate that a parameter is optional
        # Here in python 3 we can use the optional this way: Optional[type]
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data is not None: n_test = len(test_data)
        n: int = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches:list[list[tuple[np.ndarray, np.ndarray]]] = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float) -> None:
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b: list[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        nabla_w: list[np.ndarray] = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x:np.ndarray, y:np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b: list[np.ndarray] = [np.zeros(b.shape) for b in self.biases]
        nabla_w: list[np.ndarray] = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation: np.ndarray = x
        # activations is a list of arrays, where each array is the activations of the neurons in that layer
        activations: list[np.ndarray] = [x] # list to store all the activations, layer by layer
        zs: list[np.ndarray] = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z: np.ndarray = np.dot(w, activation) + b

            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta: np.ndarray = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z: np.ndarray = zs[-l]
            sp: np.ndarray = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data: list[tuple[np.ndarray, int]], test_results: list[tuple[int, int]] = None) -> int:
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        for (x, y) in test_data:
            arr: np.ndarray = self.feedforward(x)
            if test_results is not None:
                test_results.append((arr, y))
            print("================")
            print(arr)
            print(y)
            print("================")

        # [(0 ,0), 0]
        # [(0 ,1), 1]
        # [(1 ,0), 1]
        # [(1 ,0), 0]

        # test_results: list[tuple[int, int]] = [(np.argmax(self.feedforward(x)), y)
        #                for (x, y) in test_data]
        test_results: list[tuple[int, int]] = [(self.feedforward(x), y)
                        for (x, y) in test_data]

        epsilon = 0.01
        a = sum(
            int(np.abs(x - y) < epsilon)
            for (x, y) in test_results
        )
        return a

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the vector of partial derivatives partial C_x
        partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def persist_results(json_file: str, weights: list[np.ndarray], biases: list[np.ndarray], test_results: list[tuple[int, int]], epochs: int) -> None:
    # create directory if it does not exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    import json
    data = {
        "weights": [w.tolist() for w in weights],
        "biases": [b.tolist() for b in biases],
        "test_results": {
            "actual": [x.tolist() for x, y in test_results],
            "expected" : [y.tolist() for x, y in test_results]
        },
        "epochs": epochs
    }
    with open(json_file, "w") as f:
        # dump with indentation
        json.dump(data, f, indent=4)


def logic_xor():

    X_logical = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
    y_selected = np.array([[0], [1], [1], [0]])
    test_data = list(zip(X_logical, y_selected))

    # Se crea una red neuronal con 2 neuronas de entrada, 2 neuronas en la capa
    # oculta y 1 neurona de salida.
    net = MultilayerPerceptron([2, 2, 1])

    # Se entrena la red neuronal con los datos de entrada y salida esperada
    # definidos anteriormente. Se utilizan 1000 épocas y un tamaño de mini-lote
    # de 4.

    epochs = 1000

    net.fit(list(zip(X_logical, y_selected)), epochs, 4, 8) # learning rate is divided by the mini_batch_update
    net.fit(test_data, 1000, 4, 8) # learning rate is divided by the mini_batch_update

    # Se evalúa la red neuronal con los datos de entrada y salida esperada
    # definidos anteriormente.

    test_results = []

    test_data = list(zip(X_logical, y_selected))
    print(f"Accuracy: {net.evaluate(test_data, test_results)}")

    # Se guardan los pesos y sesgos de la red neuronal en un archivo JSON.
    weights = net.weights
    biases = net.biases

    persist_results(f"{RESULTS_DIR}/{R_XOR_JSON}", weights, biases, test_results, epochs)


def numberIdentifier():
    # Neuronas de entrada tiene 35 neuronas
    # Una neurona de output, par o impar

    x = np.array([
    # 0
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 1, 1,
     1, 0, 1, 0, 1,
     1, 1, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    # 1
    [0, 0, 1, 0, 0,
     0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 1, 0],
    # 2
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     0, 0, 0, 0, 1,
     0, 0, 1, 1, 0,
     0, 1, 0, 0, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    # 3
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     0, 0, 0, 0, 1,
     0, 1, 1, 1, 0,
     0, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    # 4
    [0, 0, 0, 1, 0,
     0, 0, 1, 1, 0,
     0, 1, 0, 1, 0,
     1, 0, 0, 1, 0,
     1, 1, 1, 1, 1,
     0, 0, 0, 1, 0,
     0, 0, 0, 1, 0],
    # 5
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 0,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 0,
     0, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    # 6
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    # 7
    [1, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 0, 0, 0,
     1, 0, 0, 0, 0],
    # 8
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    # 9
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 1,
     0, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
    ])

    # Output data (even = 0, odd = 1)
    y = np.array([
        [0],  # 0 is even
        [1],  # 1 is odd
        [0],  # 2 is even
        [1],  # 3 is odd
        [0],  # 4 is even
        [1],  # 5 is odd
        [0],  # 6 is even
        [1],  # 7 is odd
        [0],  # 8 is even
        [1],  # 9 is odd
    ])

    test_data = list(zip(x, y))

    net = MultilayerPerceptron([35, 10, 1])
    net.fit(test_data, 1000, 4, 5) # learning rate is divided by the mini_batch_update

    print(f"Accuracy: {net.evaluate(test_data)}")
    return 1


def parse_to_matrices(file_path: str) -> np.ndarray:
    """ Parse a file with several digits represented as 7x5 1s and 0s into a list of matrices """
    # first we load the grid
    grid = []
    with open(file_path, 'r') as file:
        for line in file:
            # consider the 0's and 1's as integers and that they are separated by spaces
            grid.append([int(x) for x in line.split()])
    # now we have the grid with the digits, we need to split it into 7x5 matrices

    # we need to split the grid into 7x5 matrices
    matrices = []
    for i in range(0, len(grid), 7):
        matrix = grid[i:i+7]
        matrices.append(matrix)
    return matrices


# determinar si un número es par/impar,
# utilizando el conjunto de datos presente en el archivo
# "TP3-ej3-digitos.txt" (con dígitos de 7 x 5 dimensiones)
# Es decir, dígitos representados con 0s y 1s, donde cada
# dígito es de 7 x 5 dimensiones.
def parity():
    # fitst we need to parse the data
    #selected: list = parse_to_matrices('../loading_data/TP3-ej3-digitos.txt')
    # i need to wrap the data into a 1D array

    digits = [
        # 0
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 1, 1],
         [1, 0, 1, 0, 1],
         [1, 1, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # 1
        [[0, 0, 1, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0]],
        # 2
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 1, 1, 0],
         [0, 1, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1]],
        # 3
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [0, 0, 0, 0, 1],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # 4
        [[0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0]],
        # 5
        [[1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # 6
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # 7
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        # 8
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
        # 9
        [[0, 1, 1, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 1],
         [0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [0, 1, 1, 1, 0]],
    ]

    # Flatten each 7x5 matrix into a 35-element vector
    selected = np.array([np.array(digit).flatten().reshape(-1, 1) for digit in digits])

    # Output data (even = 0, odd = 1)
    expected = np.array([
        [0],  # 0 is even
        [1],  # 1 is odd
        [0],  # 2 is even
        [1],  # 3 is odd
        [0],  # 4 is even
        [1],  # 5 is odd
        [0],  # 6 is even
        [1],  # 7 is odd
        [0],  # 8 is even
        [1],  # 9 is odd
    ])
    

    multilayerPerceptron = MultilayerPerceptron([35, 10, 1])

    multilayerPerceptron.fit(list(zip(selected, expected)), 20000, 4, 5)

    test_data = list(zip(selected, expected))
    print(f"Accuracy: {multilayerPerceptron.evaluate(test_data)}")



if __name__ == "__main__":
    parity()


