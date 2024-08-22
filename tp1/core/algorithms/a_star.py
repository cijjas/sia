import heapq
from core.models.node import Node
from core.heuristics import Heuristic
from core.heuristics import heuristic_combinator

# Basado en
# https://en.wikipedia.org/wiki/A*_search_algorithm

def a_star(start_node=Node, heuristics=[]):
    frontier = []
    expanded_nodes = 0
    g_score = {start_node: 0} # costo de llegar a ese nodo, sin tener en cuenta la heuristica
    f_score = {start_node: 0} # suma del g_score con la heuristica
    # explored = set()

    heapq.heappush(frontier, (0, start_node))

    while frontier:
        # heapq agarra por default el primer elemento de la tupla para comparar
        _, current_node = heapq.heappop(frontier) # Pop the smallest item off the heap, maintaining the heap invariant.

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        expanded_nodes += 1
        # explored.add(current_node)

        for child in current_node.get_children():
            tentative_g_score = g_score[current_node] + 1 # asumiendo uniform cost
            if child not in g_score or tentative_g_score < g_score.get(child, float('inf')): # si el nodo no esta en g_score, devuelve inf
                g_score[child] = tentative_g_score
                f_score[child] = tentative_g_score + heuristic_combinator(child, heuristics)
                heapq.heappush(frontier, (f_score[child], child))


    return None, expanded_nodes, len(frontier)