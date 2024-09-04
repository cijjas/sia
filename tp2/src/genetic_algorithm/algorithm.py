# seteas el algoritmo
# basicamente setea todos los hyperparametros del algoritmo genetico
# recibe las config, crea poblacion, manager del algoritmo

from genetic_algorithm.population import Population
from utils.time_manager import TimeManager

def run_genetic_algorithm(config, fitness_func, time_manager: TimeManager):


    population = Population(
        config['initial_population'],
        fitness_func,
        config['selection'],
        config['operators']['crossover'],
        config['operators']['mutation'],
        config['termination_criteria']
        )
    

    while not population.has_converged() and not time_manager.time_is_up():
        population.evolve()