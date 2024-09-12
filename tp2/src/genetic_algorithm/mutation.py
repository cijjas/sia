# GEN UNIFORME
# MULTIGEN UNIFORME

import random
from genetic_algorithm.classes.genotype import Genotype
from genetic_algorithm.classes.individual import Individual
from utils.normalizer import normalizer
from genetic_algorithm.classes.hyperparameters import Mutator
import math

def mutation_operation(individuals, mutation:Mutator, generation:int):
    mutation_method = mutation.method
    if individuals is None or len(individuals) == 0:
        return individuals
    total_points = individuals[0].genes.get_total_points()
    for individual in individuals:
        if mutation_method == "gen":
            gen_mutation(individual, total_points, mutation, generation)
        elif mutation_method == "multigen":
            multigen_mutation(individual, total_points, mutation, generation)
        elif mutation_method == "multigen_limited":
            multigen_limited_mutation(individual, total_points, mutation, generation)
        elif mutation_method == "complete":
            complete_mutation(individual, total_points, mutation, generation)

        normalizer(individual, total_points)

    return individuals






def mutate_gene(individual, index, total_points, distribution='uniform', dist_params=None):
    current_value = individual.genes[index]

    if index == (len(individual.genes) - 1):
        if distribution == 'uniform':
            low, high = (1.3, 2.0)
            new_value = round(random.uniform(low, high), 2)
        elif distribution == 'gaussian':
            mean, std_dev = (current_value, 0.1) if not dist_params else (dist_params.get('mean', current_value), dist_params.get('std_h', 0.1))
            new_value = round(random.gauss(mean, std_dev), 2)
            new_value = max(1.3, min(new_value, 2.0))  
        elif distribution == 'beta': # crecer
            alpha, beta = (3, 1) if not dist_params else (dist_params.get('alpha', 1), dist_params.get('beta', 1))
            new_value = round(random.betavariate(alpha, beta) * (2.0 - current_value) + current_value, 2)
            new_value = max(1.3, min(new_value, 2.0))
        elif distribution == 'gamma':# decrecer
            shape, scale = (2, 0.1) if not dist_params else (dist_params.get('shape', 2), dist_params.get('scale', 0.1))
            scale = current_value / shape if current_value > 0 else scale
            new_value = round(random.gammavariate(shape, scale) , 2)
            new_value = max(1.3, min(new_value, 2.0))  
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    else:
        if distribution == 'uniform':
            low, high = (0, total_points)
            new_value = random.uniform(low, high)
        elif distribution == 'gaussian':
            mean, std_dev = (current_value, 5) if not dist_params else (current_value, dist_params.get('std_p', 5))
            new_value = random.gauss(mean, std_dev)
            new_value = max(0, min(new_value, total_points))  
        elif distribution == 'beta': #crecer
            alpha, beta = (3, 1) if not dist_params else (dist_params.get('alpha', 1), dist_params.get('beta', 1))
            new_value = int(random.betavariate(alpha, beta) * (total_points - current_value) + current_value)
            new_value = max(0, min(new_value, total_points))
        elif distribution == 'gamma':# decrecer
            shape, scale = (2, 1) if not dist_params else (dist_params.get('shape', 2), dist_params.get('scale', 1))
            scale = current_value / shape if current_value > 0 else scale
            new_value = int(random.gammavariate(shape, scale) )
            new_value = max(0, min(new_value, total_points))  
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    individual.genes[index] = new_value


def gen_mutation(individual, total_points, mutator: Mutator, generation=None):
    mutation_rate = calculate_mutation_rate(mutator, generation)
    if random.random() <= mutation_rate:
        rand = random.randint(0, len(individual.genes) - 1)
        mutate_gene(individual, rand, total_points, mutator.distribution, mutator.distribution_params)

# Tal vez muta todos los genes
def multigen_mutation(individual, total_points, mutator: Mutator, generation=None):
    mutation_rate = calculate_mutation_rate(mutator, generation)
    for i in range(len(individual.genes)):
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points, mutator.distribution, mutator.distribution_params)

def multigen_limited_mutation(individual, total_points, mutator: Mutator, generation=None):
    mutation_rate = calculate_mutation_rate(mutator, generation)
    indices_to_mutate = random.sample(range(len(individual.genes)), min(mutator.amount, len(individual.genes)))
    for i in indices_to_mutate:
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points, mutator.distribution, mutator.distribution_params)

def complete_mutation(individual, total_points, mutator: Mutator, generation=None):
    mutation_rate = calculate_mutation_rate(mutator, generation)
    if random.random() <= mutation_rate:
        for i in range(len(individual.genes)):
            mutate_gene(individual, i, total_points, mutator.distribution, mutator.distribution_params)



def calculate_mutation_rate(mutator: Mutator, generation: int):
    if mutator.rate_method == "constant":
        return mutator.initial_rate
    elif mutator.rate_method == "exponential_decay":
        rate = mutator.initial_rate * (mutator.decay_rate ** generation)
        return max(rate, mutator.final_rate)
    elif mutator.rate_method == "sinusoidal":
        rate = mutator.final_rate + (mutator.initial_rate - mutator.final_rate) * (1 + math.sin(2 * math.pi * generation / mutator.period)) / 2 
        return rate
    return mutator.initial_rate 