from collections import deque
from core.models.node import Node

def bfs(start_node=Node):
    expanded_nodes = 0
    frontier = deque([start_node])
    explored = set()


    while frontier:
        current_node = frontier.popleft()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        expanded_nodes += 1 

        explored.add(current_node)
        for child in current_node.get_children():
            if child not in explored and child not in frontier:
                child.parent = current_node
                frontier.append(child)


    return None, expanded_nodes, len(frontier) # No solution
