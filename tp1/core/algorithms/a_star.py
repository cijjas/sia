import heapq
from core.models.node import Node
from core.heuristics import Heuristic
from core.heuristics import heuristic_combinator

# Basado en
# https://en.wikipedia.org/wiki/A*_search_algorithm

def a_star(start_node=Node, heuristics=[]):
    open_set = [] # expanded nodes o frontier
    expanded_nodes = 0
    g_score = {start_node: 0} # mapa de nodos a g_score
    f_score = {start_node: 0} 
    visited = set()

    heapq.heappush(open_set, (f_score[start_node], start_node))

    while open_set:
        # heapq agarra por default el primer elemento de la tupla para comparar
        _, current_node = heapq.heappop(open_set) # Pop the smallest item off the heap, maintaining the heap invariant.

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(open_set)

        visited.add(current_node)
        for child in current_node.get_children():
            tentative_g_score = g_score[current_node] + 1 # asumiendo uniform cost
            if child not in g_score or tentative_g_score < g_score.get(child, float('inf')): # si el nodo no esta en g_score, devuelve inf
                child.parent = current_node
                g_score[child] = tentative_g_score
                f_score[child] = tentative_g_score + heuristic_combinator(child, heuristics)
                heapq.heappush(open_set, (f_score[child], child))

        expanded_nodes += 1

    return None, expanded_nodes, len(open_set)