import tensorflow as tf
from models.mlp.network_2 import MultilayerPerceptron
import numpy as np
from utils.config import Config
import sys
import pygame
import sklearn.metrics as metrics

# Pygame initialization
pygame.init()

# Screen dimensions and colors
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 450
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (0, 128, 255)
BUTTON_HOVER_COLOR = (100, 149, 237)

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

    net.fit(
        training_data=training_data, 
        epochs=config.epochs, 
        mini_batch_size=config.mini_batch_size, 
        eta=config.learning_rate, 
    )

    # Measure accuracy with sklearn
    y_pred = np.array([np.argmax(net.feedforward(x)) for x, _ in test_data])
    y_true = np.array([np.argmax(y) for _, y in test_data])
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    
    return net, raw_x_test, raw_y_test

# Function to draw buttons
def draw_button(screen, text, x, y, width, height, active_color, inactive_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x + width > mouse[0] > x and y + height > mouse[1] > y:
        pygame.draw.rect(screen, active_color, (x, y, width, height))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(screen, inactive_color, (x, y, width, height))

    font = pygame.font.SysFont(None, 36)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=((x + (width / 2)), (y + (height / 2))))
    screen.blit(text_surface, text_rect)

# Function to display number and prediction using pygame
def visualize_predictions(net, x_test, y_test):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MNIST Predictor")

    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()

    index = [0]  # Using list to pass by reference

    def next_image():
        index[0] = (index[0] + 1) % len(x_test)

    def prev_image():
        index[0] = (index[0] - 1) % len(x_test)

    running = True
    while running:
        screen.fill(WHITE)

        # Display the input number
        number_image = pygame.surfarray.make_surface(x_test[index[0]].reshape(28, 28) * 255)
        screen.blit(pygame.transform.scale(number_image, (200, 200)), (50, 50))

        # Make a prediction with the neural network
        input_vector = x_test[index[0]].reshape(784, 1)
        prediction = np.argmax(net.feedforward(input_vector))

        # Show the predicted output
        text = font.render(f'Prediction: {prediction}', True, BLACK)
        screen.blit(text, (50, 300))

        # Draw buttons for next and previous images
        draw_button(screen, "Next", 170, 350, 100, 50, BUTTON_HOVER_COLOR, BUTTON_COLOR, next_image)
        draw_button(screen, "Previous", 30, 350, 100, 50, BUTTON_HOVER_COLOR, BUTTON_COLOR, prev_image)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

def main():
    # You can specify the size of the smaller dataset here
    small_set_size = 10000  # For example, train with only 1000 samples
    net, x_test, y_test = mnist_classifier(small_set_size)
    visualize_predictions(net, x_test, y_test)

if __name__ == "__main__":
    main()
