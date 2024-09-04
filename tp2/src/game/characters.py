import math
import numpy as np
from abc import ABC, abstractmethod

class Character(ABC):
    def __init__(self, height, strength=0, dexterity=0, intelligence=0, vigor=0, constitution=0):
        self.height = height
        self.strength_points = strength
        self.dexterity_points = dexterity
        self.intelligence_points = intelligence
        self.vigor_points = vigor
        self.constitution_points = constitution

        # Create temporary lists for each attribute's points
        points = [self.strength_points, self.dexterity_points, self.intelligence_points, self.vigor_points, self.constitution_points]
        
        # Apply numpy.tanh to the list of points
        tanh_values = np.tanh(0.01 * np.array(points))

        # Store the results in the attributes dictionary
        self.attributes = {
            "strength": 100 * tanh_values[0],
            "dexterity": tanh_values[1],
            "intelligence": 0.6 * tanh_values[2],
            "vigor": tanh_values[3],
            "constitution": 100 * tanh_values[4]
        }

        self.strength_total = self.attributes["strength"]
        self.dexterity_total = self.attributes["dexterity"]
        self.intelligence_total = self.attributes["intelligence"]
        self.vigor_total = self.attributes["vigor"]
        self.constitution_total = self.attributes["constitution"]

        self.atm = 0.5 - (3 * self.height - 5)**4 + (3 * self.height - 5)**2 + self.height / 2
        self.dem = 2 + (3 * self.height - 5)**4 - (3 * self.height - 5)**2 - self.height / 2

        self.attack = (self.dexterity_total + self.intelligence_total) * self.strength_total * self.atm
        self.defense = (self.vigor_total + self.intelligence_total) * self.constitution_total * self.dem

    def get_attributes(self):
        return self.attributes
    
    def set_attributes(self, attributes: dict, height=None):
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

        self.attributes = {
            "strength": self.strength_total,
            "dexterity": self.dexterity_total,
            "intelligence": self.intelligence_total,
            "vigor": self.vigor_total,
            "constitution": self.constitution_total
        }

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
