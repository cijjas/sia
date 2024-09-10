
from genetic_algorithm.classes.population import Population

from utils.time_manager import TimeManager

from genetic_algorithm.classes.individual import Individual

import numpy as np

from genetic_algorithm.classes.hyperparameters import Hyperparameters

import csv

import time

def create_individuals(size, total_points):

    individuals = []
    for _ in range(size):

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

  

def run_genetic_algorithm(config:Hyperparameters, fitness_func, time_manager: TimeManager,points: int, character: str):

    

    population = Population(

    initial_population=create_individuals(config.population_size, points),

    fitness_func=fitness_func,

    config=config,

    character=character

    )

    csv_file_path = '../output/population_evolution.csv'

    

    with open(csv_file_path, mode='w', newline='') as file:

        csv_writer = csv.writer(file)

        

        headers = ['strength', 'dexterity', 'intelligence', 'vigor', 'constitution', 'height', 'character', 'fitness', 'age']

        csv_writer.writerow(headers)

        while not population.has_converged() and not time_manager.time_is_up():

            current_population = population.get_lines()

            for individual in current_population:

                csv_writer.writerow(individual)

            population.evolve()

            best_individual = population.best_individual

        

        return best_individual