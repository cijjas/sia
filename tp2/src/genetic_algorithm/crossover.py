# Cruce de un punto
# Cruce de dos puntos
# Cruce uniforme
# Cruce anular
import random
from genetic_algorithm.population import Individual

def crossover_operation(parents, config, generation):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i].genes
        parent2 = parents[i + 1].genes
        child1, child2 = crossover((parent1, parent2), config['type'])
        offspring.extend([
            Individual(child1, generation),
            Individual(child2, generation)
        ])
    return offspring

def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = (
        parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    )
    child2 = (
        parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    )
    return child1, child2

def uniform_crossover(parent1, parent2, crossover_prob=0.5):
    child1, child2 = parent1[:], parent2[:]
    for i in range(len(parent1)):
        if random.random() < crossover_prob:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def annular_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    length = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:point + length] + parent1[point + length:]
    child2 = parent2[:point] + parent1[point:point + length] + parent2[point + length:]
    return child1, child2

def crossover(parents, method="single_point"):
    if method == "single_point":
        return single_point_crossover(*parents)
    elif method == "two_point":
        return two_point_crossover(*parents)
    elif method == "uniform":
        return uniform_crossover(*parents, crossover_prob=0.5)
    elif method == "annular":
        return annular_crossover(*parents)
    else:
        raise ValueError(f"Unknown crossover method: {method}")