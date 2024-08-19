import heapq
from core.node import Node
from core.heuristics import Heuristic

# Basado en 
# https://en.wikipedia.org/wiki/A*_search_algorithm


def a_star(start_node=Node, heuristics=None):
    open_set = [] # expanded nodes
    expanded_nodes = 0
    g_score = {start_node: 0} # mapa de nodos a g_score
    f_score = {start_node: max(heuristic(start_node.state) for heuristic in heuristics)}

    heapq.heappush(open_set, (f_score[start_node], start_node))

    while open_set:
        # heapq agarra por default el primer elemento de la tupla para comparar
        _, current_node = heapq.heappop(open_set) # Pop the smallest item off the heap, maintaining the heap invariant.
        
        expanded_nodes += 1
        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(open_set)

        for child in current_node.get_children():
            tentative_g_score = g_score[current_node] + 1 # asumiendo uniform cost
            if tentative_g_score < g_score.get(child, float('inf')): # si el nodo no esta en g_score, devuelve inf
                g_score[child] = tentative_g_score
                f_score[child] = tentative_g_score + max(heuristic(child.state) for heuristic in heuristics) 
                heapq.heappush(open_set, (f_score[child], child))

    return None, expanded_nodes, len(open_set)