import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from perceptron.perceptron_simple import PerceptronSimple

# Function to plot the current state of the Perceptron's training
def plot_perceptron(weights, X, y, epoch, config, grid_range=3):
    fig, ax = plt.subplots()
    ax.set_xlim([-grid_range, grid_range])
    ax.set_ylim([-grid_range, grid_range])
    ax.grid(True)

    # Plot data points
    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Class {class_value}')
    
    # Plot decision boundary using current weights
    if weights[2] != 0:
        x_values = np.linspace(-grid_range, grid_range, 100)
        slope = -(weights[1] / weights[2])
        intercept = -(weights[0] / weights[2])
        ax.plot(x_values, x_values * slope + intercept, 'k')
    
    # Include hyperparameter information in the plot
    hyperparams_info = f"LR: {config.get('learning_rate', 0.01)}, Epochs: {config.get('epochs', 1)}"
    ax.set_title(f'Epoch: {epoch+1} - {hyperparams_info}')
    ax.legend()

    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def main():
    config_file = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as file:
                config_file = json.load(file)
        except Exception as e:
            print(f"Failed to read configuration file: {e}")
            sys.exit(1)

    X_logical = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    outputs = {
        'AND': np.array([-1, -1, -1, 1]),
        'XOR': np.array([-1, 1, 1, -1])
    }
    y_selected = outputs.get(config_file.get('function', 'AND'))

    perceptron = PerceptronSimple(
        num_inputs=X_logical.shape[1],
        learning_rate=config_file.get('learning_rate', 0.01),
        epsilon=config_file.get('epsilon', 1e-5),
        threshold=config_file.get('threshold', 0.0),
    )

    num_epochs = config_file.get('epochs', 50)
    perceptron.fit(X_logical, y_selected, num_epochs)

    with imageio.get_writer('perceptron_training.gif', mode='I', duration=0.5) as writer:
        for epoch, weights in enumerate(perceptron.weights_history):
            img_data = plot_perceptron(weights, X_logical, y_selected, epoch, config_file)
            image = imageio.imread(img_data)
            writer.append_data(image)

if __name__ == "__main__":
    main()
