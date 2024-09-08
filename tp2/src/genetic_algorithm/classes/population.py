# Seteas population

import random
from genetic_algorithm.crossover import crossover_operation
from genetic_algorithm.selection import combined_selection
from genetic_algorithm.mutation import mutation_operation
from genetic_algorithm.classes.individual import Individual
from genetic_algorithm.classes.genotype import Genotype
from typing import List
from utils.genetic_config import GAConfig

def get_best_individual(individuals: List[Individual])->Individual:
    return max(individuals, key=lambda ind: ind.get_fitness())

def get_worst_individual(individuals: List[Individual])->Individual:
    return min(individuals, key=lambda ind: ind.get_fitness())

class Population:
    def __init__(self, initial_population, fitness_func, config:GAConfig, character): # genes_pool: lista de todos los genes posibles
        if initial_population is None or len(initial_population) == 0:
            raise ValueError("Initial population must not be empty")

        self.fitness_func = fitness_func
        self.individuals = [
                Individual(Genotype(**genes), 0, character)
                for genes in initial_population
            ]
        
        self.config = config
        # _----------
        self.evaluate_population()
        self.best_individual = get_best_individual(self.individuals)
        self.best_fitness_age = 0
        self.generation = 0

    def evaluate_population(self):
        for individual in self.individuals:
            individual.calculate_fitness(self.fitness_func)

    def select(self):
        selected_parents = combined_selection(
            self.individuals,
            self.config.parents_selection_methods,
            self.config.selection_rate,
            self.generation
        )
        return selected_parents

    def replace(self, new_kids):
        survivors = combined_selection(
            self.individuals,
            self.config.replacements_selection_methods,
            1 - self.config.selection_rate,
            self.generation
        )
        while (len(survivors) + len(new_kids) < len(self.individuals)):
            best = get_best_individual(self.individuals)
            survivors.append(best)
        while (len(survivors) + len(new_kids) > len(self.individuals)):
            worst = get_worst_individual(survivors)
            survivors.remove(worst)

        self.individuals = survivors + new_kids

    def crossover(self, parents, generation)->list:
        # Aplicar operación de cruce según la configuración
        offspring = crossover_operation(parents, self.config.crossover_method, generation)
        return offspring

    def mutate(self, offspring: list[Individual])->list: 
        # Aplicar mutaciones según la configuración
        mutated_offspring = mutation_operation(offspring, self.config.mutation_method, self.config.mutation_rate)
        return mutated_offspring

    def grow_older(self):
        """ Every Individual in the population grows older by 1 """
        for individual in self.individuals:
            individual.age += 1

    def update_best_individual(self):
        new_best_individual = get_best_individual(self.individuals)
        if self.best_individual is None or new_best_individual.get_fitness() > self.best_individual.get_fitness():
            self.best_individual = new_best_individual
            self.best_fitness_age = 0
        else:
            self.best_fitness_age += 1


    def evolve(self):
        self.evaluate_population()  # Make sure to correct spelling if it was intended as 'evaluate'
        # print('----------------')
        # print('Individuals evaluation:')
        # print('----------------')
        # print(len(self.individuals))
        # for individual in self.individuals:
            # print(individual)

        parents = self.select()
        # print('----------------')
        # print('Parents selected for crossover:')
        # print('----------------')
        # print(len(parents))
        # for parent in parents:
            # print(parent)

        offspring = self.crossover(parents, self.generation)
        # print('----------------')
        # print('Offspring resulted from crossover:')
        # print('----------------')
        # print(len(offspring))
        # for child in offspring:
        #     print(child)

        mutated_offspring = self.mutate(offspring)
        # print('----------------')
        # print('Offspring after mutations:')
        # print('----------------')
        # for mutant in mutated_offspring:
        #     print(mutant)

        self.replace(mutated_offspring)
        # print('----------------')
        # print('Individuals after replacement:')
        # print('----------------')
        # print(len(self.individuals))
        # for individual in self.individuals:
        #     print(individual)


        self.evaluate_population()
        self.generation += 1
        self.grow_older()
        self.update_best_individual()


    def get_percentage_of_elder_individuals(self, age: int)->float:
        """ Returns the portion of individuals with age greater or equal to the given age """
        elder_individuals = [individual for individual in self.individuals if individual.age >= age]
        return len(elder_individuals) / len(self.individuals)

    def has_converged(self):

        max_generations = self.config.termination_criteria.max_generations
        if max_generations is not None and self.generation >= max_generations:
            print('Max generations criteria reached')
            return True

        portion = self.config.termination_criteria.structure_portion
        if portion is not None:
            generations = self.config.termination_criteria.structure_generations

            if generations is not None:
                if self.get_percentage_of_elder_individuals(generations) >= portion:
                    print('Structure criteria reached')
                    return True

        content_generation_amount = self.config.termination_criteria.content
        if content_generation_amount is not None:
            if self.best_fitness_age >= content_generation_amount:
                print('Content criteria reached')
                return True
            return False

        desired_fitness = self.config.termination_criteria.desired_fitness
        if desired_fitness is not None:
            if self.best_individual.get_fitness() >= desired_fitness:
                print('Desired fitness criteria reached')
                return True
            return False

        return False

    def __str__(self) -> str:
        return f"Population:\n" + "\n".join(map(str, self.individuals))