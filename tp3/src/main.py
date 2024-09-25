# Import the necessary library
import numpy as np
import sys
import json
from perceptron import *
#from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
 
def main():
    config_file:dict = dict()
    if len(sys.argv) == 2:
        # Read the configuration file as a json
        config_file = json.load(open(sys.argv[1]))
        

    # Define inputs for the AND function
    X_logical = np.array([[-1, -1],
                [1, -1],
                [-1, 1],
                [1, 1]])

    # Define the target/expected output for the AND function
    y_AND = np.array([-1, -1, -1, 1])
    y_XOR = np.array([-1, 1, 1, -1])

    # NOTA
    # Para el AND la funcion converge muy rapido, lo que tiene sentido porque al ser
    # una funcion linealmente separable, el perceptron puede encontrar un hiperplano rapido
    # El XOR en cambio nunca converge. De vuelta tiene sentido porque no existe un hiperplano
    # que separe los datos de forma lineal. Ver "XOR problem for single layer perceptron"

    y_selected = y_XOR
    
    # Initialize the Perceptron with the appropriate number of inputs
    perceptron = SimplePerceptron(num_inputs=X_logical.shape[1], learning_rate=config_file.get('learning_rate'), epsilon=config_file.get('epsilon'), threshold=config_file.get('threshold'), debug=config_file.get('debug'))
    
    # Train the Perceptron on the training data
    # here X_logical is the input data and y_selected is the target data
    perceptron.fit(X_logical, y_selected, num_epochs=config_file.get('epochs'))
    
    # Prediction
    pred = perceptron.predict(X_logical)

    # Print the predictions
    print(f'Predictions: {pred}')

    # Test accuracy
    accuracy = np.mean(pred == y_selected)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
