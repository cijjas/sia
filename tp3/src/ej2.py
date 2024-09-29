import numpy as np
import sys
import json
from perceptron.perceptron_linear import PerceptronLinear
from perceptron.perceptron_non_linear import PerceptronNonLinear
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    # Load configuration file
    config_file = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as file:
                config_file = json.load(file)
        except Exception as e:
            print(f"Failed to read configuration file: {e}")
            sys.exit(1)

    # Load the dataset
    df = pd.read_csv("../res/TP3-ej2-conjunto.csv")
    print("Read dataset")

    # Split into X and y
    X = df[['x1', 'x2', 'x3']].values
    y = df['y'].values

    # Scale inputs
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Scale outputs (between 0 and 1)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(X=y.reshape(-1, 1)).ravel()

    # Initialization parameters
    num_epochs = config_file.get('epochs', 50)
    seed = config_file.get('seed', 42)
    learning_rate = config_file.get('learning_rate', 0.01)
    epsilon = config_file.get('epsilon', 1e-5)

    # Create and train Linear Perceptron
    perceptron_linear = PerceptronLinear(seed=seed, num_features=X_scaled.shape[1], learning_rate=learning_rate, epsilon=epsilon)
    perceptron_linear.train(X_scaled, y_scaled, num_epochs)

    # Configuration for Non-Linear Perceptrons
    activations = ['tanh', 'sigmoid', 'relu']
    betas = [0.1, 0.5, 0.9, 1.0, 2.0, 5.0, 10.0]
    non_linear_perceptrons = []

    for fn in activations:
        for beta in betas:
            if fn == 'relu' and beta != 1.0:
                continue  # Skip ReLU for non-standard beta values
            perceptron = PerceptronNonLinear(
                seed=seed,
                num_features=X_scaled.shape[1],
                learning_rate=learning_rate,
                epsilon=epsilon,
                non_linear_fn=fn,
                beta=(beta if fn != 'relu' else None)  # No beta for ReLU
            )
            perceptron.train(X_scaled, y_scaled, num_epochs)
            non_linear_perceptrons.append((perceptron, fn, beta))

    # Plot the loss history
    plt.figure(figsize=(12, 6))
    plt.plot(perceptron_linear.loss_history, label='Linear Perceptron', color='black')

    # Define colors from tab20 colormap
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(activations) * len(betas)))

    # Plot Non-Linear Perceptrons
    i = 0
    for perceptron, fn, beta in non_linear_perceptrons:
        plt.plot(perceptron.loss_history, label=f'{fn.capitalize()} beta={beta}', color=colors[i])
        i += 1

    # Enhance the graph
    plt.title('Loss History of Perceptrons', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)  # Adjusted legend position
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
    plt.show()

if __name__ == "__main__":
    main()
