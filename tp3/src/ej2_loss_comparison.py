import numpy as np
import sys
import json
from models.perceptrons.perceptron_linear import PerceptronLinear
from models.perceptrons.perceptron_non_linear import PerceptronNonLinear
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# This script compares the loss history of linear and non-linear perceptrons across epochs
def main():
    # Load configuration file
    config_file = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "r") as file:
                config_file = json.load(file)
        except Exception as e:
            print(f"Failed to read configuration file: {e}")
            sys.exit(1)

    # Load the dataset
    df = pd.read_csv("../res/TP3-ej2-conjunto.csv")
    print("Read dataset")

    # Split into X and y
    X = df[["x1", "x2", "x3"]].values
    y = df["y"].values

    # Scale inputs
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale outputs (between 0 and 1)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(X=y.reshape(-1, 1)).ravel()

    # Initialization parameters
    num_epochs = config_file.get("epochs", 50)
    seed = config_file.get("seed", 42)
    learning_rate = config_file.get("learning_rate", 0.01)
    epsilon = config_file.get("epsilon", 1e-5)

    # Create and train Linear Perceptron
    perceptron_linear = PerceptronLinear(
        seed=seed,
        num_features=X_scaled.shape[1],
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
    perceptron_linear.fit(X_scaled, y_scaled, num_epochs)

    # Configuration for Non-Linear Perceptrons
    activations = ["tanh", "sigmoid", "relu"]
    betas = [0.1, 0.5, 0.9, 1.0, 2.0, 5.0, 10.0]
    non_linear_perceptrons = []

    for fn in activations:
        for beta in betas:
            if fn == "relu" and beta != 1.0:
                continue  # Skip ReLU for non-standard beta values
            perceptron = PerceptronNonLinear(
                seed=seed,
                num_features=X_scaled.shape[1],
                learning_rate=learning_rate,
                epsilon=epsilon,
                non_linear_fn=fn,
                beta=(beta if fn != "relu" else None),  # No beta for ReLU
            )
            perceptron.fit(X_scaled, y_scaled, num_epochs)
            non_linear_perceptrons.append((perceptron, fn, beta))

    # Set up three subplots: one for sigmoid, one for tanh, and one for relu
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define colors from tab20 colormap
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(betas)))

    # Define the y-ticks (more y-axis labels)
    y_ticks = np.linspace(0, 0.2, 11)  # Creates 11 y-ticks between 0 and 0.2

    # Plot Sigmoid perceptrons in the first subplot
    ax_sigmoid = axes[0]
    ax_sigmoid.plot(
        perceptron_linear.loss_history, label="Linear Perceptron", color="black"
    )

    i = 0
    for perceptron, fn, beta in non_linear_perceptrons:
        if fn == "sigmoid":
            ax_sigmoid.plot(
                perceptron.loss_history, label=f"Sigmoid beta={beta}", color=colors[i]
            )
            i += 1

    # Enhance the sigmoid graph
    ax_sigmoid.set_title("Loss History of Sigmoid Perceptrons", fontsize=14)
    ax_sigmoid.set_xlabel("Epoch", fontsize=12)
    ax_sigmoid.set_ylabel("Loss", fontsize=12)
    ax_sigmoid.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax_sigmoid.grid(True, which="both")  # Enable both major and minor grid lines
    ax_sigmoid.set_yticks(y_ticks)  # Set more y-axis labels
    ax_sigmoid.set_ylim(0, 0.2)  # Fix y-axis between 0 and 0.2

    # Plot Tanh perceptrons in the second subplot
    ax_tanh = axes[1]
    ax_tanh.plot(
        perceptron_linear.loss_history, label="Linear Perceptron", color="black"
    )

    i = 0
    for perceptron, fn, beta in non_linear_perceptrons:
        if fn == "tanh":
            ax_tanh.plot(
                perceptron.loss_history, label=f"Tanh beta={beta}", color=colors[i]
            )
            i += 1

    # Enhance the tanh graph
    ax_tanh.set_title("Loss History of Tanh Perceptrons", fontsize=14)
    ax_tanh.set_xlabel("Epoch", fontsize=12)
    ax_tanh.set_ylabel("Loss", fontsize=12)
    ax_tanh.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax_tanh.grid(True, which="both")  # Enable both major and minor grid lines
    ax_tanh.set_yticks(y_ticks)  # Set more y-axis labels
    ax_tanh.set_ylim(0, 0.2)  # Fix y-axis between 0 and 0.2

    # Plot ReLU perceptrons in the third subplot
    ax_relu = axes[2]
    ax_relu.plot(
        perceptron_linear.loss_history, label="Linear Perceptron", color="black"
    )

    i = 0
    for perceptron, fn, beta in non_linear_perceptrons:
        if fn == "relu":
            ax_relu.plot(perceptron.loss_history, label=f"ReLU", color=colors[i])
            i += 1

    # Enhance the relu graph
    ax_relu.set_title("Loss History of ReLU Perceptrons", fontsize=14)
    ax_relu.set_xlabel("Epoch", fontsize=12)
    ax_relu.set_ylabel("Loss", fontsize=12)
    ax_relu.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    ax_relu.grid(True, which="both")  # Enable both major and minor grid lines
    ax_relu.set_yticks(y_ticks)  # Set more y-axis labels
    ax_relu.set_ylim(0, 0.2)  # Fix y-axis between 0 and 0.2

    # Adjust layout and display all three subplots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
