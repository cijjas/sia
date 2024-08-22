from collections import deque
from core.models.node import Node

# Preference up, right, down, left
def bfs(start_node=Node):
    frontier = deque()
    explored = set()
    expanded_nodes = 0

    frontier.append(start_node)
    explored.add(start_node)

    while frontier:
        current_node = frontier.popleft()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        for child in current_node.get_children():
            if child not in explored:
                frontier.append(child)
                explored.add(child)

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier) # No solution
