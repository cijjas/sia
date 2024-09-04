# Seteas population

import random
from genetic_algorithm.crossover import crossover_operation
from genetic_algorithm.selection import combined_selection
from genetic_algorithm.mutation import mutation_operation


class Individual:
    
    def __init__(self, genes, generation):
        self.genes = genes
        self.fitness = None
        self.generation = generation

    def calculate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.genes)




class Population:
    def __init__(self, initial_population, fitness_func, selection_method, crossover_method, mutation_method, termination_criteria): # genes_pool: lista de todos los genes posibles
        self.fitness_func = fitness_func
        self.individuals = [
            Individual(attrs, 0) # TODO ver como los recibe si recbe el caracter y la altura etc
            for attrs in initial_population
            ]
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.termination_criteria = termination_criteria
        self.generation = 0

    def evaluete_population(self):
        for individual in self.individuals:
            individual.calculate_fitness(self.fitness_func)


    
    def select(self):

        selected_parents = combined_selection(self.individuals, self.selection_method["parents"])
        replacements = combined_selection(self.individuals, self.selection_method['replacement'])

        return selected_parents, replacements

    def crossover(self, parents):
        # Aplicar operación de cruce según la configuración
        offspring = crossover_operation(parents, self.crossover_config, self.generation + 1)
        # TODO agregar o reemplazar? 

    def mutate(self):
        # Aplicar mutaciones según la configuración
        for individual in self.individuals:
            mutation_operation(individual, self.mutation_config)

    def evolve(self):
        self.evaluete_population()
        parents = self.select()
        self.crossover(parents)
        self.mutate()
        self.generation += 1
    
    def has_converged(self):
        
        return False