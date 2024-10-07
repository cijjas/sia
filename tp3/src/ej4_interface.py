import sys
import pygame
import numpy as np
import matplotlib.pyplot as plt
from models.mlp.network_2 import MultilayerPerceptron
from utils.config import Config
import tensorflow as tf
from io import BytesIO
import gzip
import pickle
import numpy as np

# Constants for the window
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

LEFT_WIDTH = WINDOW_WIDTH // 3   # 1/3 of the screen width
RIGHT_WIDTH = WINDOW_WIDTH - LEFT_WIDTH  # 2/3 of the screen width

# Grid settings
GRID_SIZE = 28  # 28x28 grid
CELL_SIZE = min(RIGHT_WIDTH, WINDOW_HEIGHT) // GRID_SIZE  # Size of each cell
GRID_OFFSET_X = LEFT_WIDTH  # Start drawing grid from here
GRID_OFFSET_Y = (WINDOW_HEIGHT - (CELL_SIZE * GRID_SIZE)) // 2  # Center the grid vertically

# Colors

TEXT = (255,255,255)
BACKGROUND = (0,0,0)
BACKGROUND_2 = (15, 15, 15)
PRIMARY = (98, 149, 132)
SECONDARY = (56, 116, 120)
ACCENT = (255, 0, 0)
GRID = (50, 50, 50)
LIGHT = (234, 237, 228)

BACKGROUND_COLOR = BACKGROUND  # Black

CELL_OFF_COLOR = BACKGROUND_2  # Black
CELL_ON_COLOR = LIGHT  # White
GRID_LINE_COLOR = GRID  # Dark gray lines for the grid

BUTTON_COLOR = LIGHT  # Apple's blue color
BUTTON_HOVER_COLOR = PRIMARY

TEXT_COLOR = TEXT  # White
PREDICTION_COLOR = TEXT  # White

BOREDER_COLOR = PRIMARY  # White


# Fonts
pygame.font.init()
FONT_SIZE_SMALL = 18
FONT_SIZE_LARGE = 24

font_small = pygame.font.Font('../res/fonts/NeueHaasDisplayRoman.ttf',FONT_SIZE_SMALL)  # Larger font for algorithm buttons and map name
font_large = pygame.font.Font('../res/fonts/NeueHaasDisplayMediu.ttf',FONT_SIZE_LARGE)  # Larger font for algorithm buttons and map name
font_larger = pygame.font.Font('../res/fonts/NeueHaasDisplayMediu.ttf', 32)  # Larger font for algorithm buttons and map name
# Confidence Colors
CONFIDENCE_COLOR_VERY = (0, 255, 0)        # Green
CONFIDENCE_COLOR_SOMEWHAT = (255, 255, 0)  # Yellow
CONFIDENCE_COLOR_NOT = (255, 0, 0)         # Red



def load_mnist(path):
    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set


def prepare_mnist_data(path):
    """
    Prepares the MNIST dataset for training.
    """
    train_set, valid_set, test_set = load_mnist(path)
    x_train, y_train = train_set
    x_valid, y_valid = valid_set
    x_test, y_test = test_set

    # Convert to numpy arrays if necessary
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

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

def train_mnist_classifier(config, path):
    training_data, test_data = prepare_mnist_data(path)
    net = MultilayerPerceptron(
        seed=config.seed,
        sizes=config.topology,
        activation_function=config.activation_function,
        optimizer=config.optimizer
    )
    net.fit(
        training_data=training_data,
        epochs=config.epochs,
        mini_batch_size=config.mini_batch_size,
        eta=config.learning_rate,
    )
    return net

def apply_blur(grid_array):
    # Simple average blur using a convolution kernel
    kernel = np.array([[1/16, 1/8, 1/16],
                       [1/8,  1/4, 1/8],
                       [1/16, 1/8, 1/16]])
    # Pad the grid_array
    padded_array = np.pad(grid_array, pad_width=1, mode='constant', constant_values=0)
    blurred_array = np.zeros_like(grid_array)
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            region = padded_array[i:i+3, j:j+3]
            blurred_array[i, j] = np.sum(kernel * region)
    return blurred_array

def preprocess_grid(grid_array):
    # Apply blur to the grid_array
    blurred_grid_array = apply_blur(grid_array)
    # Flatten the grid array to create the input vector
    input_vector = blurred_grid_array.flatten().reshape(784, 1)
    return input_vector, blurred_grid_array


def create_button(rect, text, font):
    text_surface = font.render(text, True, TEXT_COLOR)  # Default text color
    # Right-align the text within the rectangle (for hover detection)
    text_rect = text_surface.get_rect(midright=rect.midright)
    return {
        'rect': rect, 
        'text_surface': text_surface, 
        'text_rect': text_rect, 
        'text': text, 
        'font': font
        }


import matplotlib.ticker as ticker

def plot_probability_distribution(probabilities):
    """
    Creates a smaller bar plot of the probability distribution of each digit (0-9).
    Returns a Pygame surface of the plot.
    """
    # Normalize the LIGHT color to [0, 1] range for Matplotlib
    light_color = [c / 255 for c in LIGHT]
    
    # Generate the plot using Matplotlib
    custom_colors = [
        [c / 255 for c in PRIMARY]   # Normalize BUTTON_COLOR
    ]
    
    fig, ax = plt.subplots(figsize=(2, 1))  # Reduced size of plot
    
    # Plot the bar chart
    ax.bar(range(10), probabilities, tick_label=range(10), color=custom_colors)
    
    # Set the labels and title
    ax.set_ylabel("Probability", fontsize=8, color=light_color)
    ax.set_title("Probability Distribution", fontsize=10, color=light_color)
    
    # Set the tick parameters color
    ax.tick_params(axis='x', colors=light_color, labelsize=8)
    ax.tick_params(axis='y', colors=light_color, labelsize=8)
    
    # Set the spines (borders) color to LIGHT
    ax.spines['bottom'].set_color(light_color)
    ax.spines['left'].set_color(light_color)
    
    # Remove the top and right spines (optional)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Format the y-axis to show two decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Save the plot to an in-memory buffer with transparent background
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close()

    # Load the image into Pygame
    plot_surface = pygame.image.load(buf)
    buf.close()
    return plot_surface



def run_pygame_interface(net):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("MNIST Digit Recognizer")
    clock = pygame.time.Clock()

    # Initialize the grid array
    grid_array = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Buttons
    button_width = 200  # Fixed button width
    button_height = 40
    button_x = LEFT_WIDTH // 2 - button_width // 2 + 25
    button_y_start = 30
    button_spacing = 5

    predict_button_rect = pygame.Rect(
        button_x, button_y_start, button_width, button_height)
    clear_button_rect = pygame.Rect(
        button_x, button_y_start + button_height + button_spacing, button_width, button_height)

    predict_button = create_button(predict_button_rect, 'Predict', font_larger)
    clear_button = create_button(clear_button_rect, 'Clear', font_larger)

    buttons = [predict_button, clear_button]

    # Processed image display parameters
    processed_image_width = 200  # Adjusted size
    processed_image_height = 200
    processed_image_x = LEFT_WIDTH // 2 - processed_image_width // 2
    processed_image_y = None  # Will be set after rendering prediction text

    # Probability distribution plot
    prob_plot_surface = None

    prediction_text = ''
    processed_image_surface = None

    while True:
        mouse_pos = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):  # Left or right mouse button
                    # Check if any button was clicked
                    for button in buttons:
                        if button['rect'].collidepoint(event.pos):
                            if button == predict_button:
                                input_vector, processed_image = preprocess_grid(grid_array)
                                output = net.feedforward(input_vector)
                                predicted_digit = np.argmax(output)
                                max_prob = np.max(output)
                                # Determine confidence level
                                if max_prob >= 0.9:
                                    confidence_msg = "Model is very confident."
                                elif max_prob >= 0.5:
                                    confidence_msg = "Model is somewhat confident."
                                else:
                                    confidence_msg = "Model is not confident."
                                prediction_text = f"Predicted {predicted_digit}\n{confidence_msg}"

                                # Rotate and flip the processed image for correct display
                                processed_image_rotated = np.rot90(processed_image, k=3)
                                processed_image_flipped = np.fliplr(processed_image_rotated)

                                # Convert processed_image to a pygame surface for display
                                processed_image_surface = pygame.surfarray.make_surface(
                                    np.repeat(processed_image_flipped[:, :, np.newaxis], 3, axis=2) * 255
                                )
                                processed_image_surface = pygame.transform.scale(
                                    processed_image_surface, (processed_image_width, processed_image_height)
                                )

                                # Generate the probability distribution plot
                                prob_plot_surface = plot_probability_distribution(output.flatten())

                            elif button == clear_button:
                                grid_array.fill(0)
                                prediction_text = ''
                                processed_image_surface = None
                                prob_plot_surface = None
                            break  # Break after handling button click
                    else:
                        # Handle grid cell setting
                        grid_x = (event.pos[0] - GRID_OFFSET_X) // CELL_SIZE
                        grid_y = (event.pos[1] - GRID_OFFSET_Y) // CELL_SIZE
                        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                            if event.button == 1:  # Left click
                                grid_array[grid_y, grid_x] = 1.0  # Set cell to 'on'
                            elif event.button == 3:  # Right click
                                grid_array[grid_y, grid_x] = 0.0  # Set cell to 'off'

            elif event.type == pygame.MOUSEMOTION:
                if mouse_buttons[0] or mouse_buttons[2]:  # Left or right button is held down
                    mouse_pos = pygame.mouse.get_pos()
                    grid_x = (mouse_pos[0] - GRID_OFFSET_X) // CELL_SIZE
                    grid_y = (mouse_pos[1] - GRID_OFFSET_Y) // CELL_SIZE
                    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                        if mouse_buttons[0]:  # Left button held down
                            grid_array[grid_y, grid_x] = 1.0  # Set cell to 'on'
                        elif mouse_buttons[2]:  # Right button held down
                            grid_array[grid_y, grid_x] = 0.0  # Set cell to 'off'

        # Draw the grid
        screen.fill(BACKGROUND_COLOR)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                cell_rect = pygame.Rect(
                    GRID_OFFSET_X + col * CELL_SIZE,
                    GRID_OFFSET_Y + row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                color = CELL_ON_COLOR if grid_array[row, col] > 0 else CELL_OFF_COLOR
                pygame.draw.rect(screen, color, cell_rect)
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(screen, GRID_LINE_COLOR,
                             (GRID_OFFSET_X + i * CELL_SIZE, GRID_OFFSET_Y),
                             (GRID_OFFSET_X + i * CELL_SIZE, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE))
            # Horizontal lines
            pygame.draw.line(screen, GRID_LINE_COLOR,
                             (GRID_OFFSET_X, GRID_OFFSET_Y + i * CELL_SIZE),
                             (GRID_OFFSET_X + GRID_SIZE * CELL_SIZE, GRID_OFFSET_Y + i * CELL_SIZE))

        # Draw buttons with rounded rectangles
        # Draw buttons with right-aligned text
        for button in buttons:
            # Change text color on hover
            if button['rect'].collidepoint(mouse_pos):
                text_surface = button['font'].render(button['text'], True, BUTTON_HOVER_COLOR)  # Change text color on hover
            else:
                text_surface = button['font'].render(button['text'], True, BUTTON_COLOR)  # Default text color

            # Right-align the text inside the button rect
            text_rect = text_surface.get_rect(midright=button['rect'].midright)
            screen.blit(text_surface, text_rect)


        pygame.draw.line(screen, BOREDER_COLOR, (0, clear_button_rect.bottom + 20), (LEFT_WIDTH, clear_button_rect.bottom + 20), 1)
        
        # Calculate prediction text position below buttons
        prediction_text_y = clear_button_rect.bottom + 40

        processed_image_y = clear_button_rect.bottom + 40

        if processed_image_surface:
            processed_image_rect = screen.blit(processed_image_surface, (processed_image_x, processed_image_y))
            
            border_thickness = 2  # Adjust thickness to your liking
            pygame.draw.rect(screen, BOREDER_COLOR, processed_image_rect.inflate(border_thickness * 2, border_thickness * 2), border_thickness)

            # Label for the processed image (tag in the top-left corner)
            processed_image_label = font_small.render("Input Image", True, PREDICTION_COLOR)
            label_padding = 5  # Small padding for the tag
            label_rect = processed_image_label.get_rect(topleft=(processed_image_rect.left + label_padding, processed_image_rect.top + label_padding))
            
            # Draw the label on the top-left corner as a small tag
            pygame.draw.rect(screen, BOREDER_COLOR, label_rect.inflate(10, 5))  # Optional: draw background for the tag
            screen.blit(processed_image_label, label_rect)

            # Draw the probability distribution plot below the processed image
            prob_plot_y = processed_image_y + processed_image_height + 20
            if prob_plot_surface:
                screen.blit(prob_plot_surface, (processed_image_x-30, prob_plot_y))


        # Calculate prediction text position below the probability plot
        if prob_plot_surface:
            prediction_text_y = prob_plot_y + 150  # Or any suitable offset
        else:
            prediction_text_y = processed_image_y + processed_image_height + 20  # Fallback position

        # Render prediction text
        # Render prediction text right-justified
        # Render prediction text
        if prediction_text:
            lines = prediction_text.split('\n')
            right_align_x = LEFT_WIDTH - 30  # Right margin, adjust as needed
            for i, line in enumerate(lines):
                if line.startswith("Model is"):
                    # Split the line into words
                    words = line.split()
                    word_surfaces = []
                    total_width = 0
                    for word in words:
                        # Determine the color based on the word
                        if word in ["very", "somewhat", "not"]:
                            if word == "very":
                                color = CONFIDENCE_COLOR_VERY
                            elif word == "somewhat":
                                color = CONFIDENCE_COLOR_SOMEWHAT
                            elif word == "not":
                                color = CONFIDENCE_COLOR_NOT
                        else:
                            color = PREDICTION_COLOR  # Default color
                        word_surface = font_small.render(word + ' ', True, color)
                        word_width = word_surface.get_width()
                        word_surfaces.append((word_surface, word_width))
                        total_width += word_width
                    # Compute the starting x position for right alignment
                    x = right_align_x - total_width
                    y = prediction_text_y + i * (FONT_SIZE_LARGE + 5)
                    # Render each word
                    for word_surface, word_width in word_surfaces:
                        word_rect = word_surface.get_rect()
                        word_rect.topleft = (x, y)
                        screen.blit(word_surface, word_rect)
                        x += word_width  # Move x to the right by the word's width
                else:
                    # Regular rendering for other lines
                    text_surface = font_large.render(line, True, PREDICTION_COLOR)
                    text_rect = text_surface.get_rect(
                        topright=(right_align_x, prediction_text_y + i * (FONT_SIZE_LARGE + 5)))
                    screen.blit(text_surface, text_rect)


        pygame.display.flip()
        clock.tick(120)

import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file> [model_path]")
        sys.exit(1)

    # Load the configuration
    config = Config().read_config(sys.argv[1])

    # Get model path from CLI or use default
    model_path = sys.argv[2] if len(sys.argv) > 2 else "store/adam_95.npz"

    # Check if the model exists
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        net = MultilayerPerceptron.load_model(
            model_path,
            config.activation_function,
            config.optimizer
        )
    else:
        print(f"Model not found at {model_path}, starting training...")
        net = train_mnist_classifier(config, "../res/mnist.pkl.gz")
        net.save_model(model_path)

    # Run the interface
    run_pygame_interface(net)
if __name__ == "__main__":
    main()

