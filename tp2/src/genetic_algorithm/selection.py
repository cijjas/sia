# BOLTZMANN
# TORNEOS (Ambas versiones)

import random 
import math


# agarra una lista de individuos y selecciona de acuerdo a la configuración
def combined_selection(individuals, selection_config, survival_rate, generation):
    selected = []
    individuals_size = len(individuals) * survival_rate
    for config in selection_config:
        num_selected = int(config['weight'] * individuals_size) #FIXME ver si castea bien
        method = config['method']
        if method == "deterministic_tournament":
            selected.extend(deterministic_tournament_selection(individuals, num_selected, config['params']['tournament_size']))
        elif method == "probabilistic_tournament":
            selected.extend(probabilistic_tournament_selection(individuals, num_selected, config['params']['threshold']))
        elif method == "roulette":
            selected.extend(roulette_wheel_selection(individuals, num_selected))
        elif method == "elite":
            selected.extend(elite_selection(individuals, num_selected))
        elif method == "ranking":
            selected.extend(ranking_selection(individuals, num_selected))
        elif method == "universal":
            selected.extend(universal_selection(individuals, num_selected))
        elif method == "boltzmann":
            selected.extend(boltzmann_selection(individuals, num_selected, config['params']['t_0'], config['params']['t_C'], config['params']['k'], generation))
        else:
            raise ValueError(f"Unknown selection method: {method}")

    return selected


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

def boltzmann_selection(individuals, num_selected, t_0, t_C, k , generation):
    temperature = t_0 + (t_C - t_0) * math.exp(-k * generation) 
    
    avg_fitness = sum(math.exp(ind.fitness / temperature) for ind in individuals) / len(individuals)
    exp_values = [math.exp(ind.fitness / temperature) / avg_fitness  for ind in individuals]
    
    selected = random.choices(individuals, weights=exp_values, k=num_selected)
    return selected