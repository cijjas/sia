from collections import deque
from core.node import Node

def bfs(start_node=Node):
    # Initialize Fr and Ex
    expanded_nodes = 0
    frontier = deque([start_node])
    explored = set()


    while frontier:
        current_node = frontier.popleft()
        
        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes
        
        if current_node not in explored:
            expanded_nodes += 1
            explored.add(current_node)

            if current_node.is_dead_end():
                continue

            for child in current_node.get_children():
                if child not in explored:
                    frontier.append(child)


    return None, expanded_nodes
