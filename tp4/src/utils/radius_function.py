import numpy as np


EXPONENTIAL_DECAY_RADIUS = 'exponential_decay'
CONSTANT_RADIUS = 'constant'

def exponential_decay_radius(radius, iteration, max_iterations) -> float:
    return radius * np.exp(-iteration / max_iterations)

def constant_radius(radius, iteration, max_iterations) -> float:
    return radius


def str_to_radius_function(radius_function) -> callable:
    if radius_function == EXPONENTIAL_DECAY_RADIUS:
        return exponential_decay_radius
    elif radius_function == CONSTANT_RADIUS:
        return constant_radius
    else:
        raise ValueError(f'Invalid radius function: {radius_function}')

