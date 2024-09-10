
from genetic_algorithm.classes.genotype import Genotype
import numpy as np

def eve(character, genotype:Genotype):

    strength, dexterity, intelligence, vigor, constitution, height = genotype.as_array()

    total_strength = 100 * np.tanh(0.01 * strength)
    total_dexterity = np.tanh(0.01 * dexterity)
    total_intelligence = 0.6 * np.tanh(0.01 * intelligence)
    total_vigor = np.tanh(0.01 * vigor)
    total_constitution = 100 * np.tanh(0.01 * constitution)

    atm = 0.5 - (3*height - 5)**4 + (3*height - 5)**2 + height/2
    dem = 2 + (3*height - 5)**4 - (3*height - 5)**2 - height/2

    attack = (total_dexterity + total_intelligence) * total_strength * atm
    defense = (total_vigor + total_intelligence) * total_constitution * dem


    performances = {
        'warrior': 0.6 * attack + 0.4 * defense,
        'archer': 0.9 * attack + 0.1 * defense,
        'guardian': 0.1 * attack + 0.9 * defense,
        'mage': 0.8 * attack + 0.2 * defense
    }
    
    return performances[character]