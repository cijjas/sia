# Cruce de un punto
# Cruce de dos puntos
# Cruce uniforme
# Cruce anular
import random
from genetic_algorithm.classes.individual import Individual
from genetic_algorithm.classes.genotype import Genotype
from utils.normalizer import normalizer



def circular_indexing(arr, index):
    return arr[(index % len(arr))]

def crossover_operation(parents, crossover_method, generation, crossover_rate) -> list:
    offspring = []
    num_parents = len(parents)

    
    even_parents = parents
    if num_parents % 2 != 0:
        even_parents += [parents[0]] # agregar el mejor para que sea par

    for i in range(0, len(even_parents), 2):
        parent1 = even_parents[i].genes.as_array()
        parent2 = even_parents[i + 1].genes.as_array()
        total_sum = even_parents[i].genes.get_total_points()
        child1, child2 = select_crossover((parent1, parent2), crossover_method, crossover_rate)
        ind1 = Individual(Genotype(*child1), generation+1, even_parents[i].character)
        ind2 = Individual(Genotype(*child2), generation+1, even_parents[i + 1].character)
        normalizer(ind1, total_sum) # ¿QUÉ HACE NORMALIZER?
        normalizer(ind2, total_sum)
        offspring.extend([
            ind1,
            ind2
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
    length = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Randomly select crossover points
    cross_points = random.sample(range(length), 2)
    cross_points.sort()  # Ensure first point is less than second

    # Perform annular crossover
    for i in range(cross_points[0], cross_points[1]):
        child1[i], child2[i] = child2[i], child1[i]

    return child1, child2



def select_crossover(parents, method="single_point", crossover_rate=0.5):
    if method == "single_point":
        return single_point_crossover(*parents)
    elif method == "two_point":
        return two_point_crossover(*parents)
    elif method == "uniform":
        return uniform_crossover(*parents, crossover_rate)
    elif method == "annular":
        return annular_crossover(*parents)
