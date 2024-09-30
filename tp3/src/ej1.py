import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from io import BytesIO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from perceptron.perceptron_simple import PerceptronSimple

def transform_features(X):
    z = X[:, 0] * X[:, 1]
    return np.column_stack((X, z))

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_decision_boundary(weights, X, y, epoch, config):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2], label=f'Class {class_value}', depthshade=True)

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    w1, w2, w3 = weights[1], weights[2], weights[3]
    b = weights[0]

    if w3 != 0:
        zz = -(w1 * xx1 + w2 * xx2 + b) / w3
    else:
        zz = np.zeros_like(xx1)

    ax.plot_surface(xx1, xx2, zz, alpha=0.5, color='green')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Z = X1 * X2')

    ax.set_title(f'Epoch {epoch + 1} - 3D Decision Boundary')
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_2d_projected_decision_boundary(weights, X, y, epoch, config, grid_range=3):
    fig, ax = plt.subplots()
    ax.set_xlim([-grid_range, grid_range])
    ax.set_ylim([-grid_range, grid_range])
    ax.grid(True)

    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Class {class_value}')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))

    z = xx1 * xx2
    grid = np.c_[xx1.ravel(), xx2.ravel(), z.ravel()]

    z_values = np.dot(grid, weights[1:]) + weights[0]
    z_values = np.where(z_values >= 0, 1, -1)

    z_values = z_values.reshape(xx1.shape)

    ax.contourf(xx1, xx2, z_values, alpha=0.3, cmap=plt.cm.Paired)

    ax.set_title(f'Epoch {epoch + 1} - 2D Decision Boundary')
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_perceptron(weights, X, y, epoch, config, grid_range=3):
    fig, ax = plt.subplots()
    ax.set_xlim([-grid_range, grid_range])
    ax.set_ylim([-grid_range, grid_range])
    ax.grid(True)

    for class_value in np.unique(y):
        row_ix = np.where(y == class_value)
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=f'Class {class_value}')
    
    if weights[3] != 0:
        x_values = np.linspace(-grid_range, grid_range, 100)
        slope = -(weights[1] / weights[2])
        intercept = -(weights[0] / weights[2])
        ax.plot(x_values, x_values * slope + intercept, 'k')
    
    # Include hyperparameter information in the plot
    hyperparams_info = f"$\\eta$: {config.get('learning_rate', 0.01)}, Epochs: {config.get('epochs', 1)}"
    ax.set_title(f'Epoch: {epoch+1} - {hyperparams_info}')
    ax.legend()

    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def evaluate_classification(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"Classification Report for {title}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def cross_validate(X, y, config, n_splits=2):
    print(f"Performing {n_splits}-Fold Cross Validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True)
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        perceptron = PerceptronSimple(
            seed=config.get('seed', 42),
            num_features=X_train.shape[1],
            learning_rate=config.get('learning_rate', 0.01),
            epsilon=config.get('epsilon', 1e-5),
        )

        perceptron.train(X_train, y_train, config.get('epochs', 50))
        y_pred = perceptron.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, average='binary'))
        recall_list.append(recall_score(y_test, y_pred, average='binary'))
        f1_list.append(f1_score(y_test, y_pred, average='binary'))

    print("\nCross-Validation Results:")
    print(f"Accuracy: {np.mean(accuracy_list):.2f} (+/- {np.std(accuracy_list):.2f})")
    print(f"Precision: {np.mean(precision_list):.2f} (+/- {np.std(precision_list):.2f})")
    print(f"Recall: {np.mean(recall_list):.2f} (+/- {np.std(recall_list):.2f})")
    print(f"F1 Score: {np.mean(f1_list):.2f} (+/- {np.std(f1_list):.2f})")

def main():
    config_file = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r') as file:
                config_file = json.load(file)
        except Exception as e:
            print(f"Failed to read configuration file: {e}")
            sys.exit(1)

    X_logical = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    outputs = {
        'AND': np.array([-1, -1, -1, 1]),
        'XOR': np.array([-1, 1, 1, -1])
    }
    
    selected_function = config_file.get('function', 'XOR')
    y_selected = outputs.get(selected_function)

    X_transformed = transform_features(X_logical)

    cross_validate(X_transformed, y_selected, config_file)

    perceptron = PerceptronSimple(
        seed=config_file.get('seed', 42),
        num_features=X_transformed.shape[1],
        learning_rate=config_file.get('learning_rate', 0.01),
        epsilon=config_file.get('epsilon', 1e-5),
    )

    num_epochs = config_file.get('epochs', 50)
    perceptron.train(X_transformed, y_selected, num_epochs)

    y_pred = perceptron.predict(X_transformed)

    evaluate_classification(y_selected, y_pred, selected_function)

    with imageio.get_writer('perceptron_training_2d_projected.gif', mode='I', duration=0.5) as writer_2d:
        for epoch, weights in enumerate(perceptron.weights_history):
            img_data = plot_2d_projected_decision_boundary(weights, X_logical, y_selected, epoch, config_file)
            image = imageio.imread(img_data)
            writer_2d.append_data(image)

    with imageio.get_writer('perceptron_training_3d_boundary.gif', mode='I', duration=0.5) as writer_3d:
        for epoch, weights in enumerate(perceptron.weights_history):
            img_data2 = plot_3d_decision_boundary(weights, X_transformed, y_selected, epoch, config_file)
            image2 = imageio.imread(img_data2)
            writer_3d.append_data(image2)



if __name__ == "__main__":
    main()
