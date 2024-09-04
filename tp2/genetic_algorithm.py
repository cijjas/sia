from abc import ABC, abstractmethod
import random
import math

class GeneticAlgorithm(ABC):
    def __init__(self, population, fitness_func, mutation_rate=0.01):
        """
        Inicializa el algoritmo genético con la población inicial y la función de fitness.
        
        :param population: Lista inicial de individuos en la población.
        :param fitness_func: Función que evalúa el fitness de un individuo.
        :param mutation_rate: Tasa de mutación para los individuos.
        """
        self.population = population
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate

    @abstractmethod
    def selection(self, selection_size):
        """
        Método abstracto para realizar la selección de individuos.
        
        :param selection_size: Número de individuos a seleccionar.
        :return: Lista de individuos seleccionados.
        """
        pass

    def evolve(self, generations, selection_size):
        """
        Evoluciona la población a lo largo de un número de generaciones.
        
        :param generations: Número de generaciones a evolucionar.
        :param selection_size: Tamaño de la selección en cada generación.
        :return: La población final tras evolucionar.
        """
        for _ in range(generations):
            selected_individuals = self.selection(selection_size)
            offspring = self.crossover(selected_individuals)
            self.population = self.mutate(offspring)
        return self.population

    def crossover(self, selected_individuals):
        """
        Realiza el cruce entre los individuos seleccionados para generar descendencia.
        
        :param selected_individuals: Lista de individuos seleccionados para el cruce.
        :return: Lista de descendientes generados.
        """
        offspring = []
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[(i + 1) % len(selected_individuals)]
            child1, child2 = self.single_point_crossover(parent1, parent2)
            offspring.extend([child1, child2])
        return offspring

    def single_point_crossover(self, parent1, parent2):
        """
        Realiza un cruce de un solo punto entre dos padres.
        
        :param parent1: Primer padre.
        :param parent2: Segundo padre.
        :return: Dos hijos resultantes del cruce.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, offspring):
        """
        Aplica mutaciones a la descendencia según la tasa de mutación.
        
        :param offspring: Lista de descendientes generados.
        :return: Lista de descendientes tras la mutación.
        """
        for individual in offspring:
            if random.random() < self.mutation_rate:
                mutation_point = random.randint(0, len(individual) - 1)
                individual[mutation_point] = self.mutate_gene(individual[mutation_point])
        return offspring

    def mutate_gene(self, gene):
        """
        Aplica una mutación a un gen individual.
        
        :param gene: Gen a mutar.
        :return: Gen mutado.
        """
        return random.uniform(0, 1)

class EliteSelection(GeneticAlgorithm):
    def selection(self, selection_size):
        """
        Algoritmo de selección élite: selecciona los mejores individuos de la población.
        
        :param selection_size: Número de individuos a seleccionar.
        :return: Lista de individuos seleccionados.
        """
        population_size = len(self.population)
        sorted_population = sorted(self.population, key=self.fitness_func, reverse=True)
        selected_individuals = []

        for i in range(selection_size):
            n_i = math.ceil((selection_size - i) / population_size)
            selected_individuals.extend([sorted_population[i]] * n_i)

        return selected_individuals[:selection_size]

class RouletteWheelSelection(GeneticAlgorithm):
    def selection(self, selection_size):
        """
        Algoritmo de selección por ruleta: selecciona individuos proporcionalmente a su fitness.
        
        :param selection_size: Número de individuos a seleccionar.
        :return: Lista de individuos seleccionados.
        """
        population_size = len(self.population)
        fitness_values = [self.fitness_func(individual) for individual in self.population]
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]

        selected_individuals = []
        for _ in range(selection_size):
            r = random.random()
            cumulative_probability = 0
            for i, probability in enumerate(probabilities):
                cumulative_probability += probability
                if r <= cumulative_probability:
                    selected_individuals.append(self.population[i])
                    break

        return selected_individuals
