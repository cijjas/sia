import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from io import BytesIO
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold
from models.perceptrons.perceptron_simple import PerceptronSimple


def plot_2d_projected_decision_boundary(weights, X, y, epoch, config, grid_range=3):
    fig, ax = plt.subplots()
    ax.set_xlim([-grid_range, grid_range])
    ax.set_ylim([-grid_range, grid_range])
    ax.grid(True)

    # Scatter plot for the data points
    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=f"Class {class_value}")

    # Create a mesh grid for plotting the decision boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1)
    )

    # Calculate the decision boundary (Z values)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    z_values = np.dot(grid, weights[1:]) + weights[0]
    z_values = np.where(z_values >= 0, 1, -1)
    z_values = z_values.reshape(xx1.shape)

    # Plot the decision boundary and color the regions
    ax.contourf(xx1, xx2, z_values, alpha=0.3, cmap=plt.cm.Paired)

    # Plot the linear decision boundary line
    if weights[2] != 0:
        slope = -(weights[1] / weights[2])
        intercept = -(weights[0] / weights[2])
        x_values = np.linspace(x1_min, x1_max, 100)
        ax.plot(x_values, slope * x_values + intercept, "k--")

    # Include hyperparameter information in the plot
    hyperparams_info = f"$\\eta$: {config.get('learning_rate', 0.01)}, Epochs: {config.get('epochs', 1)}"
    ax.set_title(f"Epoch: {epoch+1} - {hyperparams_info}")
    ax.legend()

    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


# The rest of your code remains unchanged


def main():
    config_file = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "r") as file:
                config_file = json.load(file)
        except Exception as e:
            print(f"Failed to read configuration file: {e}")
            sys.exit(1)

    X_logical = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    outputs = {"AND": np.array([-1, -1, -1, 1]), "XOR": np.array([-1, 1, 1, -1])}

    selected_function = config_file.get("function", "XOR")
    y_selected = outputs.get(selected_function)

    perceptron = PerceptronSimple(
        seed=config_file.get("seed", 42),
        num_features=X_logical.shape[1],
        learning_rate=config_file.get("learning_rate", 0.01),
        epsilon=config_file.get("epsilon", 1e-5),
    )

    num_epochs = config_file.get("epochs", 50)
    perceptron.fit(X_logical, y_selected, num_epochs)

    y_pred = perceptron.predict(X_logical)

    with imageio.get_writer(
        "./output/ej1/perceptron_training_2d_projected.gif", mode="I", duration=0.5
    ) as writer_2d:
        for epoch, weights in enumerate(perceptron.weights_history):
            img_data = plot_2d_projected_decision_boundary(
                weights, X_logical, y_selected, epoch, config_file
            )
            image = imageio.imread(img_data)
            writer_2d.append_data(image)


if __name__ == "__main__":
    main()
