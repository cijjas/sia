import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, weights, biases, nabla_w, nabla_b):
        pass

class MiniBatchGradientDescent(Optimizer):
    def update(self, weights, biases, nabla_w, nabla_b, learning_rate, mini_batch_size):
        updated_weights = [w - (learning_rate / mini_batch_size) * gw for w, gw in zip(weights, nabla_w)]
        updated_biases = [b - (learning_rate / mini_batch_size) * gb for b, gb in zip(biases, nabla_b)]
        return updated_weights, updated_biases

class MiniBatchMomentum(Optimizer):
    def __init__(self, beta):
        self.beta = beta
        self.vw = None # velocity for weights
        self.vb = None # velocity for biases

    def update(self, weights, biases, nabla_w, nabla_b, learning_rate, mini_batch_size):
        if self.vw is None:
            self.vw = [np.zeros_like(w) for w in weights]
            self.vb = [np.zeros_like(b) for b in biases]

        self.vw = [self.beta * vw + (1 - self.beta) * gw for vw, gw in zip(self.vw, nabla_w)]
        self.vb = [self.beta * vb + (1 - self.beta) * gb for vb, gb in zip(self.vb, nabla_b)]

        updated_weights = [w - (learning_rate / mini_batch_size) * vw for w, vw in zip(weights, self.vw)]
        updated_biases = [b - (learning_rate / mini_batch_size) * vb for b, vb in zip(biases, self.vb)]

        return updated_weights, updated_biases


def str_to_optimizer(optimizer_str, beta=None) -> Optimizer:
    if optimizer_str == 'gradient_descent':
        return MiniBatchGradientDescent()
    elif optimizer_str == 'momentum':
        return MiniBatchMomentum(beta)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_str}")

