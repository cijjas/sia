# GEN UNIFORME
# MULTIGEN UNIFORME

import random
from genetic_algorithm.classes.genotype import Genotype
from genetic_algorithm.classes.individual import Individual
from utils.normalizer import normalizer

def mutation_operation(individuals, mutation_method, mutation_rate):

    total_points = individuals[0].genes.get_total_points()

    for individual in individuals:
        if mutation_method == "gen_uniform":
            gen_uniform_mutation(individual, mutation_rate, total_points)
        elif mutation_method == "multigen_uniform":
            multigen_uniform_mutation(individual, mutation_rate, total_points)
        elif mutation_method == "multigen_uniform_limited":
            amount = mutation_method["amount"]
            multigen_uniform_limited_mutation(individual, mutation_rate, total_points, amount)
        elif mutation_method == "complete":
            complete_mutation(individual, mutation_rate, total_points)


        normalizer(individual, total_points)

    return individuals

# Dado una mutacion a un gen dado los limites
def mutate_gene(individual, index, total_points):
    if index == (len(individual.genes) - 1):
        individual.genes[index] = random.uniform(1.3, 2.0)
    else:
        individual.genes[index] = random.uniform(0, total_points)

def mutate_gene2(individual, index, total_points, std_dev=0.1):
    current_value = individual.genes[index]

    if index == (len(individual.genes) - 1):
        new_value = random.uniform(1.3, 2.0)
    else:
        new_value = random.gauss(current_value, std_dev)
        new_value = max(0, min(new_value, total_points))

    individual.genes[index] = new_value

def mutate_gene3(individual, index, total_points, distribution='gaussian', dist_params=None):
    current_value = individual.genes[index]

    if index == (len(individual.genes) - 1):
        new_value = random.uniform(1.3, 2.0)
    else:
        if distribution == 'uniform':
            low, high = dist_params if dist_params else (0, total_points)
            new_value = random.uniform(low, high)
        elif distribution == 'gaussian':
            mean, std_dev = dist_params if dist_params else (current_value, 0.1)
            new_value = random.gauss(mean, std_dev)
            new_value = max(0, min(new_value, total_points))  # Clipping within range
        elif distribution == 'exponential':
            lam = dist_params[0] if dist_params else 1
            new_value = random.expovariate(lam) + current_value - lam
            new_value = max(0, min(new_value, total_points))  # Clipping within range
        elif distribution == 'beta':
            alpha, beta = dist_params if dist_params else (2, 2)
            new_value = random.betavariate(alpha, beta) * total_points
        elif distribution == 'gamma':
            shape, scale = dist_params if dist_params else (2, 1)
            new_value = random.gammavariate(shape, scale) + current_value - (shape * scale)
            new_value = max(0, min(new_value, total_points))  # Clipping within range
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    individual.genes[index] = new_value


# Elige un gen random y tal vez lo muta
def gen_uniform_mutation(individual, mutation_rate, total_points):
    if random.random() <= mutation_rate:
        rand = random.randint(0, len(individual.genes) - 1)
        mutate_gene(individual, rand, total_points)

# Tal vez muta todos los genes
def multigen_uniform_mutation(individual, mutation_rate, total_points):
    for i in range(len(individual.genes)):
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points)


def multigen_uniform_limited_mutation(individual, mutation_rate, total_points, n):
    indices_to_mutate = random.sample(range(len(individual.genes)), min(n, len(individual.genes)))
    for i in indices_to_mutate:
        if random.random() <= mutation_rate:
            mutate_gene(individual, i, total_points)

def complete_mutation(individual, mutation_rate, total_points):
    if random.random() <= mutation_rate:
        for i in range(len(individual.genes)):
            mutate_gene(individual, i, total_points)
