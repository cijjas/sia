import sys
import pygame
import numpy as np
from models.mlp.network_2 import MultilayerPerceptron
from utils.config import Config
import tensorflow as tf

# Constants for the drawing window
DRAW_AREA_SIZE = 280  # Drawing area size (pixels)
WINDOW_WIDTH = 1000  # Total window width (pixels)
WINDOW_HEIGHT = DRAW_AREA_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BRUSH_RADIUS = 12  # Brush size
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
TEXT_COLOR = (255, 255, 255)

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

def train_mnist_classifier(config):
    
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
        eta=config.learning_rate,
        epsilon=config.epsilon,
    )
    return net

def preprocess_drawing(surface):
    
    # Get the pixel data from the surface
    image = pygame.transform.scale(surface, (28, 28))
    # Convert to a numpy array and normalize
    image_array = pygame.surfarray.array3d(image)
    # Convert to grayscale
    image_array = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    image_array = image_array / 255.0
    # Flatten the array to a vector
    input_vector = image_array.reshape(784, 1)
    return input_vector

def create_button(rect, text, font):
    """
    Creates a button as a dictionary containing its rect, text, and surface.

    Parameters:
    - rect: A pygame.Rect defining the button's position and size.
    - text: The text to display on the button.
    - font: The font to use for the button text.

    Returns:
    - dict: A dictionary representing the button.
    """
    text_surface = font.render(text, True, TEXT_COLOR)
    text_rect = text_surface.get_rect(center=rect.center)
    return {'rect': rect, 'text_surface': text_surface, 'text_rect': text_rect}

def run_pygame_interface(net):
   
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Draw a digit and click 'Predict'")
    clock = pygame.time.Clock()
    drawing_surface = pygame.Surface((DRAW_AREA_SIZE, DRAW_AREA_SIZE))
    drawing_surface.fill(BLACK)
    is_drawing = False

    # Fonts
    font_small = pygame.font.SysFont(None, 24)
    font_large = pygame.font.SysFont(None, 36)

    # Buttons
    button_width = WINDOW_WIDTH - DRAW_AREA_SIZE - 20
    button_height = 50
    button_x = DRAW_AREA_SIZE + 10
    button_y_start = 50
    button_spacing = 20

    predict_button_rect = pygame.Rect(
        button_x, button_y_start, button_width, button_height)
    clear_button_rect = pygame.Rect(
        button_x, button_y_start + button_height + button_spacing, button_width, button_height)
    quit_button_rect = pygame.Rect(
        button_x, button_y_start + 2 * (button_height + button_spacing), button_width, button_height)

    predict_button = create_button(predict_button_rect, 'Predict', font_large)
    clear_button = create_button(clear_button_rect, 'Clear', font_large)
    quit_button = create_button(quit_button_rect, 'Quit', font_large)

    buttons = [predict_button, clear_button, quit_button]

    prediction_text = ''

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Start drawing when mouse button is pressed
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if drawing_surface.get_rect().collidepoint(event.pos):
                        is_drawing = True
                    else:
                        # Check if any button was clicked
                        for button in buttons:
                            if button['rect'].collidepoint(event.pos):
                                if button == predict_button:
                                    input_vector = preprocess_drawing(drawing_surface)
                                    output = net.feedforward(input_vector)
                                    predicted_digit = np.argmax(output)
                                    prediction_text = f"Predicted Digit: {predicted_digit}"
                                elif button == clear_button:
                                    drawing_surface.fill(BLACK)
                                    prediction_text = ''
                                elif button == quit_button:
                                    pygame.quit()
                                    sys.exit()

            # Stop drawing when mouse button is released
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_drawing = False

        # Draw on the surface
        if is_drawing:
            mouse_position = pygame.mouse.get_pos()
            # Adjust mouse position relative to drawing surface
            adjusted_position = (mouse_position[0], mouse_position[1])
            if drawing_surface.get_rect().collidepoint(adjusted_position):
                adjusted_position = (adjusted_position[0], adjusted_position[1])
                pygame.draw.circle(drawing_surface, WHITE, adjusted_position, BRUSH_RADIUS)

        # Draw the drawing surface onto the main screen
        screen.fill(BLACK)
        screen.blit(drawing_surface, (0, 0))

        # Draw buttons
        for button in buttons:
            # Change color on hover
            if button['rect'].collidepoint(mouse_pos):
                color = BUTTON_HOVER_COLOR
            else:
                color = BUTTON_COLOR
            pygame.draw.rect(screen, color, button['rect'])
            screen.blit(button['text_surface'], button['text_rect'])

        # Render prediction text
        if prediction_text:
            text_surface = font_small.render(prediction_text, True, (0, 255, 0))
            text_rect = text_surface.get_rect(
                center=(DRAW_AREA_SIZE + (WINDOW_WIDTH - DRAW_AREA_SIZE) // 2, 10))
            screen.blit(text_surface, text_rect)

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 frames per second
import cProfile

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config = Config().read_config(sys.argv[1])
    training_data, test_data = prepare_mnist_data()

    # Create an instance of your model
    net = MultilayerPerceptron(
        seed=config.seed,
        topology=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )

    # Profile the fit function with runctx
    cProfile.runctx('net.fit(training_data, config.epochs, config.mini_batch_size, config.learning_rate, config.epsilon)', globals(), locals())

if __name__ == "__main__":
    main()
