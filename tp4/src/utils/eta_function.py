import numpy as np

EXPONENTIAL_DECAY_ETA = "exponential_decay"
CONSTANT_ETA = "constant"


# Constant learning rate
def constant_eta(lr, iteration=None, max_iterations=None):
    return lr


# Exponential Decay learning rate
def exponential_decay_eta(lr, iteration, max_iterations):
    return lr * np.exp(-iteration / max_iterations)  # eta(t) = eta(0) * exp(-t/T)


def str_to_eta_function(eta_function) -> callable:
    if eta_function == EXPONENTIAL_DECAY_ETA:
        return exponential_decay_eta
    elif eta_function == CONSTANT_ETA:
        return constant_eta
    else:
        raise ValueError(f"Invalid eta function: {eta_function}")
