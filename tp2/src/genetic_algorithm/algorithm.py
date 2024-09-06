# seteas el algoritmo
# basicamente setea todos los hyperparametros del algoritmo genetico
# recibe las config, crea poblacion, manager del algoritmo

from genetic_algorithm.classes.population import Population
from utils.time_manager import TimeManager
from genetic_algorithm.classes.individual import Individual
import numpy as np

def create_individuals(size, total_points):
    individuals = []
    for _ in range(size):
        # Generate random attribute values that sum to total_points
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




def run_genetic_algorithm(config, fitness_func, time_manager: TimeManager,points: int, character: str):



    population = Population(
        create_individuals(10, points),
        fitness_func,
        config['selection'],
        config['operators']['crossover'],
        config['operators']['mutation'],
        config['termination_criteria'],
        character
    )

    population.evaluete_population()

    while not population.has_converged() and not time_manager.time_is_up():
        population.evolve()
    
    print(population)