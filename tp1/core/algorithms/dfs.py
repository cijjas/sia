from collections import deque
from core.models.node import Node

# Preference left, down, right, up
def dfs(start_node=Node):
    expanded_nodes = 0
    frontier = [start_node]  # stack
    explored = set()

    while frontier:
        current_node = frontier.pop()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        explored.add(current_node)  # marcar como explorado
        expanded_nodes += 1

        for child in current_node.get_children():
            if child not in explored and child not in frontier:
                frontier.append(child)

    return None, expanded_nodes, len(frontier)
