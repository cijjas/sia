import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

RESULTS_DIR = "output/ej3"
XOR_FILE = "xor.json"
PARITY_FILE = "parity.json"

def plot_accuracy_vs_epochs(json_file: str, png_name: str):
    # Load the JSON data
    with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
        data = json.load(f)
    
    # Extract test results
    test_results = data['test_results']
    
    # Calculate accuracy for each epoch
    accuracies = []
    for epoch_results in test_results:
        predicted = [pred for pred, _ in epoch_results]
        actual = [true for _, true in epoch_results]
        
        # Round the predicted values to the nearest integer
        predicted_rounded = [int(round(pred)) for pred in predicted]
        
        # Flatten the actual values list
        actual_flat = np.array(actual).flatten()
        
        # Calculate accuracy
        accuracy = accuracy_score(actual_flat, predicted_rounded)
        accuracies.append(accuracy)
    
    # Plot the accuracy vs epochs
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.grid(True)
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    plt.show()

if __name__ == "__main__":
    #plot_accuracy_vs_epochs(XOR_FILE, "xor_accuracy_vs_epochs.png")
    plot_accuracy_vs_epochs(PARITY_FILE, "parity_accuracy_vs_epochs.png")

