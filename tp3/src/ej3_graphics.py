import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score

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
        predicted_rounded = [int(round(pred[0])) for pred in predicted]
        
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
        plt.plot(epochs, accuracies, linestyle='-', label=filename_to_label(filename=file_name))
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory

def plot_cross_validation_accuracy_vs_epochs(json_file: str, png_name: str):
    """ This creates a plot with the accuracy vs epochs for cross-validation results."""
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
    plt.figure()
    plt.plot(epochs, accuracies, linestyle='-', label=filename_to_label(json_file))
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy vs Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory

def filename_to_label(filename:str):
    if filename.find("train") != -1:
        return "train"
    if filename.find("res") != -1:
        return "test"
    return filename.split('/')[-1]

def plot_precision_recall_curve(json_file: str, png_name: str):
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

def plot_precision(json_file: str, png_name: str):
    """ This creates a bar chart with the precision for each digit (0-9)."""
    # Load the JSON data
    with open(f'{RESULTS_DIR}/{json_file}', 'r') as f:
        data = json.load(f)
    
    # Extract test results
    test_results = data['test_results']
    
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    
    for epoch_results in test_results:
        for predicted, actual in epoch_results:
            predicted_label = np.argmax(predicted)
            actual_label = np.argmax(actual)
            predicted_labels.append(predicted_label)
            true_labels.append(actual_label)
    
    # Calculate precision for each digit (0-9)
    precisions = precision_score(true_labels, predicted_labels, average=None, labels=range(10))
    
    # Create a bar chart
    digits = range(10)
    plt.figure()
    plt.bar(digits, precisions, color='skyblue')
    
    plt.xlabel('Digit')
    plt.ylabel('Precision')
    plt.title('Precision for Each Digit (0-9)')
    plt.xticks(digits)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(f'{RESULTS_DIR}/{png_name}')
    #plt.show()
    plt.close()  # Close the figure to free up memory


def main():
    # check for arguments
    if len(sys.argv) != 2:
        print("Usage: python ej3_graphics.py <option>")
        sys.exit(1)
    if sys.argv[1] == "clean":
        plot_multi_output_accuracy_vs_epochs("clean_clean/res_digit.json", "digit_accuracy_vs_epochs_clean_clean.png") #TODO HECHO
    elif sys.argv[1] == "clean_noisy":
        several_plot_multi_output_accuracy_vs_epochs(["clean_noisy1/res_digit.json", "clean_noisy1/digit_train.json"], "digit_accuracy_vs_epochs_clean_noisy1.png")
    elif sys.argv[1] == "noisy1_noisy2":
        plot_cross_validation_accuracy_vs_epochs("noisy1_noisy2/cross_val_res_digit.json", "noisy1_noisy2_cross_val.png")
    elif sys.argv[1] == "precision":
        plot_precision("noisy1_noisy2/cross_val_res_digit.json", "noisy1_noisy2_precision.png")
    elif sys.argv[1] == "second_train_salt_pepper":
        plot_cross_validation_accuracy_vs_epochs("second_train_salt_pepper/cross_val_res_digit.json", "training_with_salt_and_pepper_cross_val.png")


if __name__ == "__main__":
    main()