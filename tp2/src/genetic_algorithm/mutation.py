# GEN UNIFORME
# MULTIGEN UNIFORME

import random
from genetic_algorithm.classes.genotype import Genotype
from genetic_algorithm.classes.individual import Individual
from utils.normalizer import normalizer
from genetic_algorithm.classes.hyperparameters import Mutator

def mutation_operation(individuals, mutation:Mutator):
    mutation_method = mutation.method
    total_points = individuals[0].genes.get_total_points()
    for individual in individuals:
        if mutation_method == "gen":
            gen_mutation(individual, mutation.rate, total_points, mutation.distribution, mutation.distribution_params)
        elif mutation_method == "multigen":
            multigen_mutation(individual, mutation.rate, total_points, mutation.distribution, mutation.distribution_params)
        elif mutation_method == "multigen_limited":
            multigen_limited_mutation(individual, mutation.rate, total_points, mutation.amount, mutation.distribution, mutation.distribution_params)
        elif mutation_method == "complete":
            complete_mutation(individual, mutation.rate, total_points, mutation.distribution, mutation.distribution_params)

        normalizer(individual, total_points)

    return individuals





def mutate_gene(individual, index, total_points, distribution='uniform', dist_params=None):
    current_value = individual.genes[index]
    
    if index == (len(individual.genes) - 1):
        new_value = random.uniform(1.3, 2.0)
    else:
        if distribution == 'uniform':
            low, high = (0, total_points)
            new_value = random.uniform(low, high)
        elif distribution == 'gaussian':
            # Default params for gaussian distribution, using current_value if mean is not provided
            mean, std_dev = (current_value, 5) if not dist_params else (dist_params.get('mean', current_value), dist_params.get('std', 5))
            new_value = random.gauss(mean, std_dev)
            new_value = max(0, min(new_value, total_points))  # Clipping within range
        elif distribution == 'exponential':
            # Default parameter for exponential distribution
            lambd = 1 if not dist_params else dist_params.get('lambda', 1)
            new_value = random.expovariate(lambd)
            new_value = max(0, min(new_value, total_points))

        elif distribution == 'beta':
            # Default parameters for beta distribution
            alpha, beta = (2, 2) if not dist_params else (dist_params.get('alpha', 2), dist_params.get('beta', 2))
            new_value = random.betavariate(alpha, beta) * total_points
        elif distribution == 'gamma':
            # Default parameters for gamma distribution
            shape, scale = (2, 1) if not dist_params else (dist_params.get('shape', 2), dist_params.get('scale', 1))
            new_value = random.gammavariate(shape, scale) + current_value - (shape * scale)
            new_value = max(0, min(new_value, total_points))  # Clipping within range
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    individual.genes[index] = new_value


# Elige un gen random y tal vez lo muta
def gen_mutation(individual, mutation_rate, total_points, distribution='uniform', dist_params=None):
    if random.random() <= mutation_rate:
        rand = random.randint(0, len(individual.genes) - 1)
        mutate_gene(individual, rand, total_points, distribution, dist_params)


# Tal vez muta todos los genes
def multigen_mutation(individual, mutation_rate, total_points, distribution='uniform', dist_params=None):
    for i in range(len(individual.genes)):
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points, distribution, dist_params)


def multigen_limited_mutation(individual, mutation_rate, total_points, n, distribution='uniform', dist_params=None):
    indices_to_mutate = random.sample(range(len(individual.genes)), min(n, len(individual.genes)))
    for i in indices_to_mutate:
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points, distribution, dist_params)

def complete_mutation(individual, mutation_rate, total_points, distribution='uniform', dist_params=None):
    if random.random() <= mutation_rate:
        for i in range(len(individual.genes)):
            mutate_gene(individual, i, total_points, distribution, dist_params)
