from core.models.node import Node
from core.models.state import State
from core.heuristics import Heuristic


def greedy_local(start_node: Node, heuristics=None):
    expanded_nodes = 0
    frontier = [start_node]
    explored = set()

    while frontier:
        current_node = frontier.pop(0)

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        if current_node not in explored:
            explored.add(current_node)

            children = current_node.get_children()
            children.sort(key=lambda child: max(heuristic(child.state) for heuristic in heuristics)) # TODO faltaba chequear que no esten en explored
            for child in children:
                if child not in explored and child not in frontier:
                    frontier.append(child)


        
        expanded_nodes += 1

    return None, expanded_nodes, len(frontier)
