import heapq
from core.node import Node


def a_star(start_node=Node, heuristi_func=None):
    open_set = []
    heapq.heappush(open_set, start_node)
    came_from = {}
    f_score = {start_node: heuristi_func}

    while open_set:
        current = heapq.heappop(open_set) # Pop the smallest item off the heap, maintaining the heap invariant.

        if current.is_goal():
            return current.get_path(), len(came_from)

        for child in current.get_children():
            tentative_g_score = g_score[current] + 1

            if child not in g_score or tentative_g_score < g_score[child]:
                came_from[child] = current
                g_score[child] = tentative_g_score
                f_score[child] = g_score[child] + heuristi_func(child.state)
                heapq.heappush(open_set, child)