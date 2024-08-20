from collections import deque
from core.models.node import Node
def bfs(start_node=Node):
    # Initialize Fr and Ex
    expanded_nodes = 0
    frontier = deque([start_node])
    explored = set()

    while frontier:
        current_node = frontier.popleft()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        expanded_nodes += 1
        for child in current_node.get_children():
            if child not in explored:
                explored.add(child)
                child.parent = current_node
                frontier.append(child)

    return None, expanded_nodes, len(frontier) # No solution
