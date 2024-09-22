# Import the necessary library
import numpy as np
import perceptron
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
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
perceptron = perceptron.Perceptron(num_inputs=X_logical.shape[1], learning_rate=0.03, debug=True)
 
# Train the Perceptron on the training data
perceptron.fit(X_logical, y_selected, num_epochs=100)
 
# Prediction
pred = perceptron.predict(X_logical)

# Print the predictions
print(f'Predictions: {pred}')

# Test accuracy
accuracy = np.mean(pred == y_selected)
print(f'Accuracy: {accuracy}')