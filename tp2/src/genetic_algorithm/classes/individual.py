
from genetic_algorithm.classes.genotype import Genotype

class Individual:

    def __init__(self, genes: Genotype, generation: int, character: str):
        self._genes = genes  # Use a private attribute to store the genes
        self.character = character
        self.fitness = None
        self.generation = generation
        self.age = 0

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, new_genes):
        self.set_genes(new_genes)

    def set_genes(self, genes):
        self.age = 0
        self._genes = genes

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.character, self.genes)

    def get_fitness(self):
        return self.fitness

    def get_generation(self):
        return self.generation

    def set_generation(self, generation):
        self.generation = generation

    def get_genes(self):
        return self.genes.as_array()

    def __str__(self):
        return f"{self.genes}, {self.character}, {self.fitness}, {self.age}"

    def get_line(self):
        # Return a list of attributes for the individual, formatted as needed for CSV output
        return self.genes.as_array() + [self.character, self.fitness, self.age]
