import random
from characters import *
from eve import EVE
from genetic_algorithm import *

class GeneticAlgorithmEngine:
    def __init__(self, population_size, generations, selection_method, eve: EVE):
        self.population_size = population_size
        self.generations = generations
        self.selection_method = selection_method
        self.eve = eve

    def initialize_population(self):
        """Genera una población inicial de personajes aleatorios."""
        population = []
        for _ in range(self.population_size):
            character_class = random.choice([Warrior, Archer, Guardian, Mage])
            character = character_class()
            attributes = {
                'strength': random.randint(0, 100),
                'dexterity': random.randint(0, 100),
                'intelligence': random.randint(0, 100),
                'vigor': random.randint(0, 100),
                'constitution': random.randint(0, 100)
            }
            character.set_attributes(attributes)
            character.set_height(random.uniform(1.3, 2.0))
            population.append(character)
        return population

    def evolve(self):
        """Evoluciona la población a través de varias generaciones."""
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Evaluar el desempeño de cada personaje
            fitness_scores = [self.eve.evaluate(char) for char in population]
            
            # Seleccionar los mejores individuos
            selected_population = self.selection_method.select(population, fitness_scores)
            
            # Generar nueva población a partir de los seleccionados
            new_population = self.generate_new_population(selected_population)
            
            # Aplicar mutación a la nueva población
            self.mutate(new_population)
            
            population = new_population
            
            print(f"Generation {generation+1}: Best Performance: {max(fitness_scores)}")
        
        # Devolver el mejor personaje de la última generación
        best_character = max(population, key=lambda char: self.eve.evaluate(char))
        return best_character
    
    def generate_new_population(self, selected_population):
        """Genera una nueva población cruzando los seleccionados."""
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected_population, 2)
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        return new_population
    
    def crossover(self, parent1: Character, parent2: Character):
        """Cruzamiento de dos personajes para generar un hijo."""
        character_class = random.choice([Warrior, Archer, Guardian, Mage])
        child = character_class()
        attributes = {
            'strength': random.choice([parent1.attributes['strength'], parent2.attributes['strength']]),
            'dexterity': random.choice([parent1.attributes['dexterity'], parent2.attributes['dexterity']]),
            'intelligence': random.choice([parent1.attributes['intelligence'], parent2.attributes['intelligence']]),
            'vigor': random.choice([parent1.attributes['vigor'], parent2.attributes['vigor']]),
            'constitution': random.choice([parent1.attributes['constitution'], parent2.attributes['constitution']])
        }
        child.set_attributes(attributes)
        child.set_height(random.choice([parent1.height, parent2.height]))
        return child
    
    # population is a list of characters
    def mutate(self, population: list[Character]):
        """Aplica mutaciones aleatorias a la población."""
        mutation_rate = 0.01
        for character in population:
            if random.random() < mutation_rate:
                attribute = random.choice(list(character.attributes.keys()))
                character.attributes[attribute] = random.randint(0, 100)


def main():

    population_size = 100
    generations = 10
    selection_method = RouletteWheelSelection()
    eve = EVE(Warrior)  # Suponiendo que estás evaluando para la clase Warrior

    selection_method = EliteSelection()
    engine = GeneticAlgorithmEngine(population_size, generations, selection_method, eve)
    best_character: Character = engine.evolve()

    print(f"Best Character: {best_character}")
    print(f"Attributes: {best_character.attributes}")
    print(f"Height: {best_character.height}")
    print(f"Attack: {best_character.attack}")
    print(f"Defense: {best_character.defense}")
    print(f"Performance: {eve.evaluate(best_character)}")
    
    engine = GeneticAlgorithmEngine(population_size, generations, selection_method, eve)
    best_character = engine.evolve()
    
    print(f"Best character: {best_character.__class__.__name__}")
    print(f"Attributes: {best_character.attributes}")
    print(f"Height: {best_character.height}")
    print(f"Attack: {best_character.attack}")
    print(f"Defense: {best_character.defense}")
    print(f"Performance: {eve.evaluate(best_character)}")


if __name__ == "__main__":
    main()
