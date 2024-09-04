# ELITE
# RULETA
# UNIVERSAL
# BOLTZMANN
# TORNEOS (Ambas versiones)
# RANKING

import random 
import math


# agarra una lista de individuos y selecciona de acuerdo a la configuración
def combined_selection(individuals, selection_config):
    selected = []
    individuals_size = len(individuals)
    for config in selection_config:
        num_selected = int(config['weight'] * individuals_size) #FIXME ver si castea bien
        method = config['method']
        if method == "tournament":
            selected.extend(tournament_selection(individuals, num_selected, config['params']['tournament_size']))
        elif method == "roulette":
            selected.extend(roulette_wheel_selection(individuals, num_selected))
        elif method == "elite":
            selected.extend(elite_selection(individuals, num_selected))
        elif method == "ranking":
            selected.extend(ranking_selection(individuals, num_selected))


def tournament_selection(individuals, num_selected, tournament_size):
    # Implementación básica del torneo
    winners = []
    for _ in range(num_selected):
        participants = random.sample(individuals, tournament_size)
        winner = max(participants, key=lambda ind: ind.fitness)
        winners.append(winner)
    return winners

def roulette_wheel_selection(individuals, num_selected):
    # Implementación básica de la ruleta
    total_fitness = sum(ind.fitness for ind in individuals)
    selection_probs = [ind.fitness / total_fitness for ind in individuals]
    chosen = random.choices(individuals, weights=selection_probs, k=num_selected)
    return chosen

def elite_selection(individuals, num_selected):
    # Selecciona los mejores según el fitness
    sorted_by_fitness = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    return sorted_by_fitness[:num_selected]

def ranking_selection(individuals, num_selected):
    # Selecciona basado en un ranking lineal de fitness
    sorted_by_fitness = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    return sorted_by_fitness[:num_selected]

