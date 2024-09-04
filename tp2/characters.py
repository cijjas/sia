import math
from abc import ABC, abstractmethod

class Character(ABC):
    def __init__(self, height, strength=0, dexterity=0, intelligence=0, vigor=0, constitution=0):
        self.height = height
        self.strength_points = strength
        self.dexterity_points = dexterity
        self.intelligence_points = intelligence
        self.vigor_points = vigor
        self.constitution_points = constitution

        self.strength_total = 100 * math.tanh(0.01 * self.strength_points)
        self.dexterity_total = math.tanh(0.01 * self.dexterity_points)
        self.intelligence_total = 0.6 * math.tanh(0.01 * self.intelligence_points)
        self.vigor_total = math.tanh(0.01 * self.vigor_points)
        self.constitution_total = 100 * math.tanh(0.01 * self.constitution_points)

        self.atm = 0.5 - (3 * self.height - 5)**4 + (3 * self.height - 5)**2 + self.height / 2
        self.dem = 2 + (3 * self.height - 5)**4 - (3 * self.height - 5)**2 - self.height / 2

        self.attack = (self.dexterity_total + self.intelligence_total) * self.strength_total * self.atm
        self.defense = (self.vigor_total + self.intelligence_total) * self.constitution_total * self.dem

    def get_attributes(self):
        return {
            "strength_total": self.strength_total,
            "dexterity_total": self.dexterity_total,
            "intelligence_total": self.intelligence_total,
            "vigor_total": self.vigor_total,
            "constitution_total": self.constitution_total
        }
    
    def set_attributes(self, attributes, height=None):
        self.strength_points = attributes.get('strength', self.strength_points)
        self.dexterity_points = attributes.get('dexterity', self.dexterity_points)
        self.intelligence_points = attributes.get('intelligence', self.intelligence_points)
        self.vigor_points = attributes.get('vigor', self.vigor_points)
        self.constitution_points = attributes.get('constitution', self.constitution_points)

        self.strength_total = 100 * math.tanh(0.01 * self.strength_points)
        self.dexterity_total = math.tanh(0.01 * self.dexterity_points)
        self.intelligence_total = 0.6 * math.tanh(0.01 * self.intelligence_points)
        self.vigor_total = math.tanh(0.01 * self.vigor_points)
        self.constitution_total = 100 * math.tanh(0.01 * self.constitution_points)

        if height is not None:
            self.height = height
            self.atm = 0.5 - (3 * self.height - 5)**4 + (3 * self.height - 5)**2 + self.height / 2
            self.dem = 2 + (3 * self.height - 5)**4 - (3 * self.height - 5)**2 - self.height / 2

        self.attack = (self.dexterity_total + self.intelligence_total) * self.strength_total * self.atm
        self.defense = (self.vigor_total + self.intelligence_total) * self.constitution_total * self.dem

    def set_height(self, height):
        self.height = height
        self.atm = 0.5 - (3 * self.height - 5)**4 + (3 * self.height - 5)**2 + self.height / 2
        self.dem = 2 + (3 * self.height - 5)**4 - (3 * self.height - 5)**2 - self.height / 2

        self.attack = (self.dexterity_total + self.intelligence_total) * self.strength_total * self.atm
        self.defense = (self.vigor_total + self.intelligence_total) * self.constitution_total * self.dem

    @abstractmethod
    def calculate_performance(self):
        """Calcula el desempeño del personaje basado en su rol (ataque y defensa)."""
        pass

class Warrior(Character):
    def calculate_performance(self):
        return 0.6 * self.attack + 0.4 * self.defense

class Archer(Character):
    def calculate_performance(self):
        return 0.9 * self.attack + 0.1 * self.defense

class Guardian(Character):
    def calculate_performance(self):
        return 0.1 * self.attack + 0.9 * self.defense

class Mage(Character):
    def calculate_performance(self):
        return 0.8 * self.attack + 0.3 * self.defense
