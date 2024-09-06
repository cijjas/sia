# GEN
# MULTIGEN
# UNIFORME
# NO_UNIFORME (OPCIONAL!!!!!!!!!!!!!!!)

import random
from abc import ABC, abstractmethod
from genetic_algorithm.classes.individual import Individual
from genetic_algorithm.classes.genotype import Genotype

""" The Mutation class is an abstract class that defines the interface for mutation operations.
    The __call__ method is the one that should be implemented by the subclasses.
    The __str__ method is implemented to return the name of the mutation operation.
"""

class Mutation(ABC):
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    @abstractmethod
    def __call__(self, genes: dict):
        pass

    def __str__(self):
        return self.__class__.__name__

class GenMutation(Mutation):
    """ GenMutation is a mutation operation that flips the value of a random gene in the genotype.
    Given that attributes must sum up to a given value, only the height attribute is flipped."""
    def __call__(self, genes: dict):
        if random.random() < self.mutation_rate:
            genes['height'] = round(random.uniform(1.3, 2.0), 2)
        return genes

class MultigenFlipMutation(Mutation):
    """ MultigenMutation is a mutation operation that flips the value of a random gene in the genotype.
    Given that attributes must sum up to a given value, only the non-height attributes are flipped."""
    def __call__(self, genes: dict):
        """ Selects two random attributes to mutate and flip their values """

        if random.random() >= self.mutation_rate:
            return genes

        # get the list of keys of the genes
        keys = list(genes.keys())
        
        # remove the height key
        keys.remove('height')
        
        # select two random keys
        key1, key2 = random.sample(keys, 2)
        
        # flip the values of the selected keys
        aux = genes[key1]
        genes[key1] = genes[key2]
        genes[key2] = aux

        return genes

class MultigenSumMutation(Mutation):
    """ MultigenSumMutation is a mutation operation that redistributes the value of two random genes in the genotype.
    Given that attributes must sum up to a given value, only the non-height attributes are modified."""
    def __call__(self, genes: dict):
        """ Selects two random attributes to mutate and redistributes their values """

        if random.random() >= self.mutation_rate:
            return genes

        # get the list of keys of the genes
        keys = list(genes.keys())
        
        # remove the height key
        keys.remove('height')
        
        # select two random keys
        key1, key2 = random.sample(keys, 2)

        # get the values of the selected keys
        value1 = genes[key1]
        value2 = genes[key2]

        # get the sum of the values
        total = value1 + value2

        # get a random value between 0 and the total
        new_value1 = random.uniform(0, total)

        # set the new values
        genes[key1] = new_value1
        genes[key2] = total - new_value1

        return genes

mutation_map = {
    "gen": GenMutation,
    "multigen_flip": MultigenFlipMutation,
    "multigen_sum": MultigenSumMutation
}

def mutation_operation(individual: Individual, mutation_dict: dict):
    """ Applies the mutation operation to the genes with the given mutation rate """
    mutation_rate = mutation_dict['rate']
    mutation_method = mutation_dict['method']

    # Get the mutation class from the map
    mutation_class = mutation_map.get(mutation_method)

    if mutation_class is None:
        raise ValueError(f"Unknown mutation method: {mutation_method}")

    # Instantiate the mutation class and apply it to the genes
    mutation = mutation_class(mutation_rate)
    genes = individual.genes.as_dict()
    mutated_genes = mutation(genes)
    individual.genes = Genotype(**mutated_genes)
    return individual
