import numpy as np
from typing import Literal

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: confusion matrix
    """
    n_classes = 2
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((y_true == i) & (y_pred == j))
    return cm

# accuracy_score: (True Positives + True Negatives) / (True Positives + False Positives + True Negatives + False Negatives)
def accuracy_score(cm: np.ndarray) -> float:
    """
    Compute accuracy score.
    :param cm: confusion matrix
    :return: accuracy score
    """
    sum = np.sum(cm)
    if sum == 0:
        return 0
    return np.sum(np.diag(cm)) / sum

# precision_score: True Positives / (True Positives + False Positives)
def precision_score(cm: np.ndarray, average: Literal['binary'] = 'binary') -> float | np.ndarray:
    """
    Compute precision score binary
    :param cm: confusion matrix
    :return: precision score
    """
    if average == 'binary':
        tp = cm[1, 1]
        fp = cm[0, 1]
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    elif average == 'micro':
        tp = np.sum(np.diag(cm))
        fp = np.sum(cm, axis=0) - np.diag(cm)
        if np.sum(tp + fp) == 0:
            return 0
        return np.sum(tp) / np.sum(tp + fp)


# recall_score: True Positives / (True Positives + False Negatives)
def recall_score(cm: np.ndarray, average: Literal['binary'] = 'binary') -> float | np.ndarray:
    """
    Compute recall score binary
    :param cm: confusion matrix
    :return: recall score
    """
    if average == 'binary':
        tp = cm[1, 1]
        fn = cm[1, 0]
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    elif average == 'micro':
        tp = np.sum(np.diag(cm))
        fn = np.sum(cm, axis=1) - np.diag(cm)
        if np.sum(tp + fn) == 0:
            return 0
        return np.sum(tp) / np.sum(tp + fn)

# f1_score: 2 * (precision * recall) / (precision + recall)
def f1_score(cm: np.ndarray, average: Literal['binary'] = 'binary') -> float | np.ndarray:
    """
    Compute f1 score binary
    :param cm: confusion matrix
    :return: f1 score
    """
    precision = precision_score(cm, average)
    recall = recall_score(cm, average)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean absolute error.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R^2 score.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: R^2 score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute median absolute error.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: median absolute error
    """
    return np.median(np.abs(y_true - y_pred))


def custom_kfold(X, y, n_splits=5, shuffle=True, random_state=42):
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))
    
    fold_sizes = np.full(n_splits, len(X) // n_splits, dtype=int)
    fold_sizes[:len(X) % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
    return folds

