import sys
import tensorflow as tf
from models.mlp.network import MultilayerPerceptron
import numpy as np
from utils.config import Config
import sklearn.metrics as metrics

# Same strucutre as the "ej4.py" file but using the "network.py" which has slight variations to allow for easier testing and metrics creation
def prepare_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    training_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_train, y_train)]
    test_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_test, y_test)]

    return training_data, test_data


def mnist_classifier(config: Config):
    training_data, test_data = prepare_mnist_data()
    net = MultilayerPerceptron(config.seed, config.topology, config.activation_function, config.optimizer)
    net.fit_with_matrix_confusion(training_data, config.epochs, config.mini_batch_size, config.learning_rate, test_data=test_data, epsilon=config.epsilon)
    accuracy = net.evaluate(test_data, epsilon=config.epsilon)
    print(f"Accuracy: {accuracy} / {len(test_data)}")

################################################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])

    mnist_classifier(config)