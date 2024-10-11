import numpy as np

EXPONENTIAL_DECAY_ETA = 'exponential_decay'
CONSTANT_ETA = 'constant'

# constant 
def constant_eta(radius, iteration, max_iterations):
    return radius


def exponential_decay_eta(lr, iteration, max_iterations):
    return lr * np.exp(-iteration / max_iterations)

def str_to_eta_function(eta_function) -> callable:
    if eta_function == EXPONENTIAL_DECAY_ETA:
        return exponential_decay_eta
    elif eta_function == CONSTANT_ETA:
        return constant_eta
    else:
        raise ValueError(f'Invalid eta function: {eta_function}')

