# GEN
# MULTIGEN
# UNIFORME
# NO_UNIFORME (OPCIONAL!!!!!!!!!!!!!!!)

import random

def gen_mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, 1)
    return individual

def multigen_mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def uniform_mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, 1)
    return individual

