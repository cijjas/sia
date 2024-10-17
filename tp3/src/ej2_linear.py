import sys
import json
from models.perceptrons.perceptron_linear import PerceptronLinear
import pandas as pd


def logistic_regression(config_file):
    # Load Config
    with open(config_file, "r") as file:
        config_file = json.load(file)

    # Load Dataset
    df = pd.read_csv("../res/TP3-ej2-conjunto.csv")
    X = df[["x1", "x2", "x3"]].values
    y = df["y"].values

    perceptron_linear = PerceptronLinear(
        seed=config_file.get("seed", 0),
        num_features=X.shape[1],
        learning_rate=config_file.get("learning_rate", 0.01),
        epsilon=config_file.get("epsilon", 1e-5),
    )

    perceptron_linear.fit(X, y, config_file.get("epochs", 100))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    logistic_regression(sys.argv[1])
