import numpy as np
import sys
import json
from models.perceptrons.perceptron_non_linear import PerceptronNonLinear
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def logistic_regression(config_file):
    # Load Config
    with open(config_file, 'r') as file:
        config_file = json.load(file)

    function = config_file.get('non_linear_fn')
    if (function == 'sigmoid'):
        scaler_sigmoid = MinMaxScaler(feature_range=(0, 1))
        X = scaler_sigmoid.fit_transform(X)
        y = scaler_sigmoid.fit_transform(y.reshape(-1, 1)).ravel()
    elif (function == 'tanh'):
        scaler_tanh = MinMaxScaler(feature_range=(-1, 1))
        X = scaler_tanh.fit_transform(X)
        y = scaler_tanh.fit_transform(y.reshape(-1, 1)).ravel()

    # Load Dataset
    df = pd.read_csv("../res/TP3-ej2-conjunto.csv")
    X = df[['x1', 'x2', 'x3']].values
    y = df['y'].values

    perceptron_non_linear = PerceptronNonLinear(
        seed=config_file.get('seed', 0),
        num_features=X.shape[1],
        learning_rate=config_file.get('learning_rate', 0.01),
        epsilon=config_file.get('epsilon', 1e-5),
        non_linear_fn=config_file.get('non_linear_fn', 'sigmoid'),
        beta=config_file.get('beta', 0.9)
    )

    perceptron_non_linear.train(X, y, config_file.get('epochs', 100))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    logistic_regression(sys.argv[1])