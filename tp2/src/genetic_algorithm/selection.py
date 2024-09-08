# BOLTZMANN
# TORNEOS (Ambas versiones)

import random
import math
from genetic_algorithm.classes.individual import Individual
from utils.genetic_config import SelectionMethod
from typing import List


def elite_selection(individuals, num_selected):
    return sorted(individuals, key=lambda x: x.fitness, reverse=True)[:num_selected]

def roulette_wheel_selection(individuals, num_selected):
    total_fitness = sum(ind.fitness for ind in individuals)
    p_i = [ind.fitness / total_fitness for ind in individuals]
    q_i = [sum(p_i[:i+1]) for i in range(len(p_i))]
    selected = []
    for _ in range(num_selected):
        r = random.uniform(0, 1)
        for i, q in enumerate(q_i):
            if r < q:
                selected.append(individuals[i])
                break

    return selected

def ranking_selection(individuals, num_selected):
    if not individuals:
        return []

    sorted_individuals = sorted(individuals, key=lambda x: x.fitness)
    n = len(individuals)
    weights = [(n - rank) / n for rank in range(n)]
    selected = random.choices(sorted_individuals, weights=weights, k=num_selected)
    return selected

def universal_selection(individuals, num_selected):
    total_fitness = sum(ind.fitness for ind in individuals)
    p_i = [ind.fitness / total_fitness for ind in individuals]
    q_i = [sum(p_i[:i+1]) for i in range(len(p_i))]
    r = random.uniform(0, 1)
    selected = []
    for j in range(num_selected):
        r_j = (r + j) / num_selected
        for i, q in enumerate(q_i):
            if r_j < q:
                selected.append(individuals[i])
                break

    return selected

def deterministic_tournament_selection(individuals, num_selected, tournament_size):
    selected = []
    for _ in range(num_selected):
        tournament = random.sample(individuals, tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        selected.append(winner)
    return selected

def probabilistic_tournament_selection(individuals, num_selected, threshold):
    selected = []
    for _ in range(num_selected):
        participants = random.sample(individuals, 2)
        r = random.uniform(0, 1)
        if r < threshold:
            winner = max(participants, key=lambda x: x.fitness)
        else:
            winner = min(participants, key=lambda x: x.fitness)
        selected.append(winner)
    return selected

def boltzmann_selection(individuals: List[Individual], num_selected, t_0, t_C, k, generation):
    temperature = t_C + (t_0 - t_C) * math.exp(-k * generation)
    avg_fitness = sum(math.exp(ind.fitness / temperature) for ind in individuals) / len(individuals)
    exp_values = [math.exp(ind.fitness / temperature) / avg_fitness for ind in individuals]
    selected = random.choices(individuals, weights=exp_values, k=num_selected)
    return selected

selection_functions = {
        "elite": elite_selection,
        "roulette": roulette_wheel_selection,
        "ranking": ranking_selection,
        "universal": universal_selection,
        "deterministic_tournament": deterministic_tournament_selection,
        "probabilistic_tournament": probabilistic_tournament_selection,
        "boltzmann": boltzmann_selection
    }

def get_selection_methods():
    return list(selection_functions.keys())

def combined_selection(individuals, selection_methods: List[SelectionMethod], survival_rate, generation) -> list:
    selected = []  # TODO puede haber repetidos y tiende a seleccionar menos que el porcentaje dado por los redondeos
    total_to_select = int(len(individuals) * survival_rate)
    
    for method in selection_methods:
        selection_size = int(method.weight * total_to_select)  # FIXME ver si castea bien
        if method.method in selection_functions:
            if method.method == "deterministic_tournament":
                selected.extend(selection_functions[method.method](individuals, selection_size, method.tournament_size))
            elif method.method == "probabilistic_tournament":
                selected.extend(selection_functions[method.method](individuals, selection_size, method.threshold))
            elif method.method == "boltzmann":
                selected.extend(selection_functions[method.method](individuals, selection_size, method.t_0, method.t_C, method.k, generation))
            else:
                selected.extend(selection_functions[method.method](individuals, selection_size))
    
    return selected
