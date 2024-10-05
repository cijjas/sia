import numpy as np
import tensorflow as tf
from models.mlp.network import MultilayerPerceptron
from utils.config import Config
import tqdm


def prepare_mnist_data():
    """
    Prepares the MNIST dataset for training.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    training_data = [(x.reshape(784, 1), y.reshape(10, 1))
                     for x, y in zip(x_train, y_train)]
    test_data = [(x.reshape(784, 1), y.reshape(10, 1))
                 for x, y in zip(x_test, y_test)]
    return training_data, test_data

def train_mnist_classifier(config: Config):
    training_data, test_data = prepare_mnist_data()
    net = MultilayerPerceptron(
        seed=config.seed,
        optimizer=config.optimizer,
        activation_function=config.activation_function,
        sizes=config.topology,
    )
    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
        eta=config.learning_rate,
        test_data=test_data,
    )
    return net, test_data

def evaluate_mnist_classifier(net, test_data):
   
    correct = 0
    total = len(test_data)
    # Evaluate each test example
    for x, y in test_data:
        output = net.feedforward(x)
        predicted_label = np.argmax(output)
        actual_label = np.argmax(y)
        print("prediction", predicted_label)
        print("actual", actual_label)
        
        if predicted_label == actual_label:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy on MNIST test data: {accuracy * 100:.2f}% - correct {correct} out of {total}")

def main():
    # Load configuration from a config file or manually set the values
    config = Config().read_config("../config/ej4.json")  # Update this with your config file path

    # Train the model
    net, test_data = train_mnist_classifier(config)

    # Evaluate the model on test data
    evaluate_mnist_classifier(net, test_data)

if __name__ == "__main__":
    main()
