from collections import deque
from core.structure.node import Node

# Basado en 
# https://en.wikipedia.org/wiki/Breadth-first_search#:~:text=Breadth%2Dfirst%20search%20(BFS),at%20the%20next%20depth%20level.

def dfs(start_node=Node):
    # Initialize Fr and Ex
    expanded_nodes = 0
    frontier = [start_node]  # stack
    explored = set()

    while frontier:
        current_node = frontier.pop()  
        
        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)
        
        expanded_nodes += 1
        explored.add(current_node)
        for child in current_node.get_children():
            if child not in explored:
                child.parent = current_node
                frontier.append(child)  # push stack

    return None, expanded_nodes, len(frontier) 
