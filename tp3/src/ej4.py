import tensorflow as tf
from models.mlp.network import MultilayerPerceptron
import numpy as np
from utils.config import Config
import sys
import sklearn.metrics as metrics

# Loading the MNIST dataset and adapting it for the neural network with a subset of the data
def prepare_mnist_data(small_set_size=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizing the input data
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

    # Converting labels to one-hot encoded format
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # If a smaller subset is requested, reduce the size of the training set
    if small_set_size is not None:
        x_train = x_train[:small_set_size]
        y_train = y_train[:small_set_size]

    # Reshaping the data so it is coherent with how the network expects it
    training_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_train, y_train)]
    test_data = [(x.reshape(784, 1), y.reshape(10, 1)) for x, y in zip(x_test, y_test)]

    return training_data, test_data, x_test, y_test

def mnist_classifier(small_set_size=None):
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])

    training_data, test_data, raw_x_test, raw_y_test = prepare_mnist_data(small_set_size)

    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    # net.load_network('persisted_net_config.json')

    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
        eta=config.learning_rate,
        epsilon=0.01
    )

    # Measure accuracy with sklearn
    y_pred = np.array([np.argmax(net.feedforward(x)) for x, _ in test_data])
    y_true = np.array([np.argmax(y) for _, y in test_data])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    net.save_network('persisted_net_config.json')

    return net, raw_x_test, raw_y_test

def main():
    # You can specify the size of the smaller dataset here
    small_set_size = 60000 # For example, train with only 1000 samples
    net, x_test, y_test = mnist_classifier(small_set_size)

if __name__ == "__main__":
    main()
