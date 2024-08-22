from collections import deque
from core.models.node import Node

# expanded nodes could be after the for in children
# cuando llega a goal no suma el nodo goal a los explorados, ni suma los expanded_nodes
# agregar a la frontier los childrens (si la estructura ignora repetidos) deberia ser equivalente a la logica actual de loopear por los children

# Preference up, right, down, left
def bfs(start_node=Node):
    expanded_nodes = 0
    frontier = deque([start_node])
    explored = set()

    while frontier:
        current_node = frontier.popleft()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        explored.add(current_node)

        for child in current_node.get_children():
            if child not in explored and child not in frontier:
                frontier.append(child)

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier) # No solution
