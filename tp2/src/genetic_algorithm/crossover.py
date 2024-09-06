# Cruce de un punto
# Cruce de dos puntos
# Cruce uniforme
# Cruce anular
import random
from genetic_algorithm.classes.individual import Individual
from genetic_algorithm.classes.genotype import Genotype

def crossover_operation(parents, config: dict, generation, fitness_function) -> list:
    offspring = []
    num_parents = len(parents)
    
    # Ensure the number of parents is even
    if num_parents % 2 != 0:
        num_parents -= 1
    
    for i in range(0, num_parents, 2):
        parent1 = parents[i].genes.as_array()
        parent2 = parents[i + 1].genes.as_array()
        child1, child2 = select_crossover((parent1, parent2), config['method'])
        ind1 = Individual(Genotype(*child1), generation+1, parents[i].character)
        ind2 = Individual(Genotype(*child2), generation+1, parents[i + 1].character)
        ind1.calculate_fitness(fitness_function)
        ind2.calculate_fitness(fitness_function)
        offspring.extend([
            ind1,
            ind2
        ])
    
    # Optionally handle the last parent if the number of parents is odd
    if len(parents) % 2 != 0:
        last_parent = parents[-1]
        offspring.append(Individual(last_parent.genes, generation, last_parent.character))
    
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

# Define a mapping from crossover method strings to crossover functions
crossover_map = {
    "single_point": single_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover,
    "annular": annular_crossover
}

def select_crossover(parents, method="single_point"):
    if method in crossover_map:
        return crossover_map[method](*parents)
    else:
        raise ValueError(f"Unknown crossover method: {method}")