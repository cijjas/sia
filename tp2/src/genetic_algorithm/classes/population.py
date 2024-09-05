# Seteas population

import random
from genetic_algorithm.crossover import crossover_operation
from genetic_algorithm.selection import combined_selection



from genetic_algorithm.classes.individual import Individual

class Population:
    def __init__(self, initial_population, fitness_func, selection_method, crossover_method, mutation_method, termination_criteria, character): # genes_pool: lista de todos los genes posibles
        self.fitness_func = fitness_func
        self.individuals = [
                Individual(genes, 0, character)
                for genes in initial_population
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
        offspring = crossover_operation(parents, self.crossover_method, self.generation + 1)
        # TODO agregar o reemplazar? 


    # TODO
    # def mutate(self):
    #     # Aplicar mutaciones según la configuración
    #     for individual in self.individuals:
    #         mutation_operation(individual, self.mutation_config)

    def evolve(self):
        self.evaluete_population()
        parents = self.select()
        self.crossover(parents)
        self.mutate()
        self.generation += 1
    
    def has_converged(self):



        max_generations = self.termination_criteria.get('max_generations', None)
        if max_generations is not None and self.generation >= max_generations:
            return True

        max_time = self.termination_criteria.get('max_time', None)
        if max_time is not None and self.generation >= max_time:
            return True

        structure = self.termination_criteria['structure']
        if structure is not None:
            portions = structure.get('portions', None)
            generations = structure.get('generations', None)
            # TODO checkear si tanta cantidad de porcion no cambio en tanta cantidad de generaciones
            return False
            
        content_generation_amount = self.termination_criteria.get('content', None)
        if content_generation_amount is not None:
            # TODO checkear si en tantas generaciones no cambio el mejor individuo
            return False

        desired_fitness = self.termination_criteria.get('desired_fitness', None)
        if desired_fitness is not None:
            # TODO checkear si el mejor individuo tiene la fitness deseada
            return False

        return False