import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

RESULTS_DIR = "output/ej3"
XOR_FILE = "xor.json"
PARITY_FILE = "parity.json"
NUM_TRAIN_PREDICTION_FILE = "digit_train.json"
NUM_TEST_PREDICTION_FILE = "res_digit.json"

def plot_single_output_accuracy_vs_epochs(json_file: str, png_name: str):
    # check if the file exists
    try:
        with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file} not found")
        return # continue with the next file
    
    # Create a new figure
    plt.figure()
    
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
    #plt.show()
    plt.close()  # Close the figure to free up memory

def plot_multi_output_accuracy_vs_epochs(json_file: str, png_name: str):
    # check if the file exists
    try:
        with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file} not found")
        return # continue with the next file

    # Create a new figure
    plt.figure()
    
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
        
        # For each output, find the index of the max value (predicted label)
        predicted_labels = [np.argmax(pred) for pred in predicted]
        actual_labels = [np.argmax(true) for true in actual]
        
        # Calculate accuracy
        accuracy = accuracy_score(actual_labels, predicted_labels)
        accuracies.append(accuracy)
    
    # Plot the accuracy vs epochs
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.grid(True)
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory

def plot_multi_output_accuracy_single_epoch(json_file: str, png_name: str):
    """ This creates a bar plot of the accuracy of each digit for a single epoch """
    # check if the file exists
    try:
        with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file} not found")
        return # continue with the next file

    # Load the JSON data
    with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
        data = json.load(f)
    
    # Extract test results
    test_results = data['test_results']
    
    # Get the last epoch's results
    last_epoch_results = test_results[-1]
    
    # Extract the predicted and actual values
    predicted = [pred for pred, _ in last_epoch_results]
    actual = [true for _, true in last_epoch_results]
    
    # For each output, find the index of the max value (predicted label)
    predicted_labels = [np.argmax(pred) for pred in predicted]
    actual_labels = [np.argmax(true) for true in actual]
    
    # Calculate accuracy for each digit
    accuracies = []
    for digit in range(10):
        # Get the indices of the actual labels that are equal to the current digit
        digit_indices = np.where(np.array(actual_labels) == digit)
        
        # Get the predicted labels for the current digit
        digit_predicted_labels = [predicted_labels[i] for i in digit_indices[0]]
        
        # Calculate accuracy for the current digit
        accuracy = accuracy_score(digit_indices[0], digit_predicted_labels)
        accuracies.append(accuracy)
    
    # Create a new figure
    plt.figure()
    
    # Plot the accuracy of each digit
    digits = range(10)
    plt.bar(digits, accuracies, color='b')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of each digit')
    plt.grid(False)
    plt.xticks(digits, [str(digit) for digit in digits])
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory

def several_plot_multi_output_accuracy_vs_epochs(files: list[str], png_name: str):
    """ This creates a single plot with the accuracy vs epochs of several files.
      Uses one color per file.
      Indicates the name of each line generated by the file name."""
    # Create a new figure
    plt.figure()
    
    # Load the JSON data
    for file in files:
        with open(f'{RESULTS_DIR}/{file}', 'r') as f:
            data = json.load(f)
        
        # Extract test results
        test_results = data['test_results']
        
        # Calculate accuracy for each epoch
        accuracies = []
        for epoch_results in test_results:
            predicted = [pred for pred, _ in epoch_results]
            actual = [true for _, true in epoch_results]
            
            # For each output, find the index of the max value (predicted label)
            predicted_labels = [np.argmax(pred) for pred in predicted]
            actual_labels = [np.argmax(true) for true in actual]
            
            # Calculate accuracy
            accuracy = accuracy_score(actual_labels, predicted_labels)
            accuracies.append(accuracy)
        
        # Plot the accuracy vs epochs
        epochs = range(1, len(accuracies) + 1)
        file_name = file.split('/')[-1]  # Get the file name
        plt.plot(epochs, accuracies, linestyle='-', label=file_name)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory

if __name__ == "__main__":
    #plot_single_output_accuracy_vs_epochs(XOR_FILE, "xor_accuracy_vs_epochs.png")
    #plot_single_output_accuracy_vs_epochs(PARITY_FILE, "parity_accuracy_vs_epochs.png")
    #plot_multi_output_accuracy_vs_epochs(NUM_TEST_PREDICTION_FILE, "digit_accuracy_vs_epochs.png")
    #plot_multi_output_accuracy_single_epoch(NUM_TRAIN_PREDICTION_FILE, "train_digit_accuracy_vs_epochs.png")
    
    #plot_multi_output_accuracy_vs_epochs("clean_clean/res_digit.json", "digit_accuracy_vs_epochs_clean_clean.png")
    #plot_multi_output_accuracy_vs_epochs("clean_noisy1/res_digit.json", "digit_accuracy_vs_epochs_clean_noisy1.png")

    several_plot_multi_output_accuracy_vs_epochs(["noisy1_noisy1/res_digit.json", "noisy1_noisy1/digit_train.json"], "digit_accuracy_vs_epochs_noisy1_noisy1.png")