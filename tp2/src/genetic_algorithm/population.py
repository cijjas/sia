# Seteas population
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def evaluate_fitness(self, fitness_func):
        self.fitness = fitness_func(self.genes)
