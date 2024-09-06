# Seteas population

import random
from genetic_algorithm.crossover import crossover_operation
from genetic_algorithm.selection import combined_selection
from genetic_algorithm.mutation import mutation_operation
from genetic_algorithm.classes.individual import Individual
from genetic_algorithm.classes.genotype import Genotype
from typing import List

class Population:
    def __init__(self, initial_population, fitness_func, selection_method, crossover_method: dict, mutation_method: dict, termination_criteria: dict, character): # genes_pool: lista de todos los genes posibles
        self.fitness_func = fitness_func
        self.individuals = [
                Individual(Genotype(**genes), 0, character)
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
        selected_parents = combined_selection(
            self.individuals, 
            self.selection_method["parents"], 
            1 - self.selection_method["selection_rate"],
            self.generation
        )
        return selected_parents
    
    def replace(self, new_kids):
        survivors = combined_selection(
            self.individuals, 
            self.selection_method['replacement'], 
            1 - self.selection_method['selection_rate'],
            self.generation
        )
        self.individuals = survivors + new_kids

    def crossover(self, parents, generation)->list:
        # Aplicar operación de cruce según la configuración
        offspring = crossover_operation(parents, self.crossover_method, generation)
        return offspring 

    def mutate(self, offspring: list[Individual])->list: # TODO: check i am recieving a list of individuals
        # Aplicar mutaciones según la configuración
        mutated_offspring = []
        for child in offspring:
            mutated_child = mutation_operation(child, self.mutation_method)
            mutated_offspring.append(mutated_child)
        return mutated_offspring

    def grow_older(self):
        """ Every Individual in the population grows older by 1 """
        for individual in self.individuals:
            individual.age += 1

    def evolve(self):
        
        # Seleccionar padres para la cruza
        parents = self.select()

        # Cruzar padres para generar hijos
        offspring = self.crossover(parents, self.generation)
        
        # Mutar hijos
        tri_eyed_kids = self.mutate(offspring)

        # Reemplazar la población actual con los hijos
        self.replace(tri_eyed_kids)

        # Evaluar la población
        self.evaluete_population()

        # Incrementar la generación
        self.generation += 1
        self.grow_older()

    
    def get_percentage_of_elder_individuals(self, age: int)->float:
        """ Returns the portion of individuals with age greater or equal to the given age """
        elder_individuals = [individual for individual in self.individuals if individual.age >= age]
        return len(elder_individuals) / len(self.individuals)

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

            if portions is not None and generations is not None:
                for portion in portions:
                    if self.get_percentage_of_elder_individuals(portion) >= generations:
                        return True
            
        content_generation_amount = self.termination_criteria.get('content', None)
        if content_generation_amount is not None:
            # TODO checkear si en tantas generaciones no cambio el mejor individuo
            return False

        desired_fitness = self.termination_criteria.get('desired_fitness', None)
        if desired_fitness is not None:
            best_individual = max(self.individuals, key=lambda ind: ind.get_fitness())
            if best_individual.get_fitness() >= desired_fitness:
            # TODO checkear si el mejor individuo tiene la fitness deseada
                return True
            return False

        return False
    
    def __str__(self) -> str:
        return f"Population:\n" + "\n".join(map(str, self.individuals))