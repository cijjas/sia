import sys
import tensorflow as tf
from models.mlp.network_2 import MultilayerPerceptron
import numpy as np
from utils.config import Config
import sklearn.metrics as metrics

# Loading the MNIST dataset and adapting it for the neural network
def prepare_mnist_data():
    # Retrieve data using the keras API
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizing the input data
    # The reshaping is to flatten out the matrix thats originally 28*28 and we want the 768 elements in one column
    # The division normalizes each pixel which has a value between 0 and 255, into a value between 0 and 1
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

    # This takes the expected output which is simply the integer and translates it into a the 10 elements version
    # So for example, if y_train[i] = 3, it replaces it with [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # Inputting y train which is an array simply does the same operation for all the values in the array
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # Reshaping the data so it is coherent with how the network expects it
    training_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_train, y_train)]
    test_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_test, y_test)]

    return training_data, test_data


def mnist_classifier(config: Config):
    training_data, test_data = prepare_mnist_data()

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

    print(f"Accuracy: {net.evaluate(test_data, epsilon=config.epsilon)} / {len(test_data)}")

################################################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])

    mnist_classifier(config)