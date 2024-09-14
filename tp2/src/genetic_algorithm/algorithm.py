from genetic_algorithm.classes.population import Population
from utils.time_manager import TimeManager
from genetic_algorithm.classes.individual import Individual
import numpy as np
from genetic_algorithm.classes.hyperparameters import Hyperparameters
import csv

def _generate_polynomial_coeffs_(min_values, max_values):
    if len(min_values) != len(max_values):
        raise ValueError("min_values and max_values must have the same length")
    random_coeffs = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in zip(min_values, max_values)])
    normalized_coeffs = random_coeffs / np.sum(random_coeffs)

    return normalized_coeffs

def create_individuals(size, total_points, seed=None):
    individuals = []
    for _ in range(size):
        if seed is not None and not seed["ignore"]:
            attributes = ["strength", "dexterity", "intelligence", "vigor", "constitution"]
            min_values = [seed[attr]["min"] for attr in attributes]
            max_values = [seed[attr]["max"] for attr in attributes]
            coeffs = _generate_polynomial_coeffs(min_values, max_values)
            distributed_points = total_points * coeffs
            print(distributed_points) if seed["debug"] else None
            attr_str, attr_dex, attr_int, attr_vig, attr_con = distributed_points
            individual = {
                "strength": int(round(attr_str)),
                "dexterity": int(round(attr_dex)),
                "intelligence": int(round(attr_int)),
                "vigor": int(round(attr_vig)),
                "constitution": int(round(attr_con)),
                "height": round(np.random.uniform(seed["height"]["min"], seed["height"]["max"]), 2)
            }

            

            individuals.append(individual)
        else:
            points = np.random.multinomial(total_points, [1/5]*5)

            individual = {
                "strength": int(points[0]),
                "dexterity": int(points[1]),
                "intelligence": int(points[2]),
                "vigor": int(points[3]),
                "constitution": int(points[4]),
                "height": round(np.random.uniform(1.3, 2.0), 2)
            }

            individuals.append(individual)

    return individuals

  

def run_genetic_algorithm(config:Hyperparameters, fitness_func, time_manager: 
                          TimeManager,points: int, character: str, initial_population = None, debug=False) -> Population:

    population = Population(
        initial_population=create_individuals(config.population_size, points, seed=initial_population),
        fitness_func=fitness_func,
        config=config,
        character=character
    )

    while not population.has_converged(show_message=debug) and not time_manager.time_is_up(show_message=debug):
        population.evolve()

    
    return population