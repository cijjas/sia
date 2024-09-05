
from genetic_algorithm.classes.genotype import Genotype
class Individual:
    
    def __init__(self, genes, generation, character):
        self.genes = Genotype(
            genes['strength'],
            genes['dexterity'],
            genes['intelligence'],
            genes['vigor'],
            genes['constitution'],
            genes['height'],
        )
        self.character = character
        self.fitness = None
        self.generation = generation

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.character, self.genes)

    def get_fitness(self):
        return self.fitness
    
    def get_generation(self):
        return self.generation
    
    def get_genes(self):
        return self.genes.as_array()
    
    def __str__(self):
        return f"Individual: {self.genes}, {self.character}, {self.fitness}, {self.generation}"

