from collections import deque
from core.models.node import Node

# Preference left, down, right, up
def dfs(start_node=Node):
    frontier = []
    explored = set()
    expanded_nodes = 0

    frontier.append(start_node)
    explored.add(start_node)

    while frontier:
        current_node = frontier.pop()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        for child in current_node.get_children():
            if child not in explored:
                frontier.append(child)
                explored.add(child)  # marcar como explorado

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier)
