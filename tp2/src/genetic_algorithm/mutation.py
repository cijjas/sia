# GEN UNIFORME
# MULTIGEN UNIFORME

import random
from genetic_algorithm.classes.genotype import Genotype
from genetic_algorithm.classes.individual import Individual

def mutation_operation(individuals, mutation_method):
    method = mutation_method["method"]
    mutation_rate = mutation_method["rate"]

    total_points = individuals[0].genes.get_total_points()

    for individual in individuals:
        if method == "gen_uniform":
            gen_uniform_mutation(individual, mutation_rate, total_points)
        elif method == "multigen_uniform":
            multigen_uniform_mutation(individual, mutation_rate, total_points)

        normalize(individual, total_points)

# Dado una mutacion a un gen dado los limites
def mutate_gene(individual, index, total_points):
    if index == (len(individual.genes) - 1):
        individual.genes[index] = random.uniform(1.3, 2.0)
    else:
        individual.genes[index] = random.uniform(0, total_points)

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


def normalize(individual, total_points):
    current_sum = individual.genes.get_total_points()

    if current_sum == total_points:
        return

    scaling_factor = total_points / current_sum
    individual.genes.attributes = [int(attr * scaling_factor) for attr in individual.genes.attributes[:-1]] + [individual.genes.attributes[-1]]

    final_sum = individual.genes.get_total_points()
    residual = total_points - final_sum

    while abs(residual) > 0 :
        random_idx = random.randint(0, len(individual.genes.attributes) - 2)
        if residual > 0:
            adjustment = min(residual, 1)
            individual.genes[random_idx] += adjustment
        else:
            adjustment = max(residual, -1)
            individual.genes[random_idx] -= adjustment

        residual = total_points - individual.genes.get_total_points()
        if individual.genes[random_idx] < 0:
            residual += individual.genes[random_idx]  # Correct the residual for setting value to zero
            individual.genes[random_idx] = 0