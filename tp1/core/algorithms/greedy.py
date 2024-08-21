from core.models.node import Node
from core.models.state import State
from core.heuristics import Heuristic


def local_greedy(start_node: Node, heuristics=None):
    """ local greedy search considers the heuristic value of most recently expanded node's children """
    expanded_nodes = 0
    frontier = [start_node]
    explored = set()

    while frontier:
        current_node = frontier.pop(0)

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        if current_node not in explored:
            explored.add(current_node)

            if current_node.is_dead_end(): # TODO franfer deadend no era una heurística
                continue

            children = current_node.get_children()
            children.sort(key=lambda child: max(heuristic(child.state) for heuristic in heuristics)) # TODO faltaba chequear que no esten en explored
            for child in children:
                if child not in explored and child not in frontier:
                    frontier.append(child)


        
        expanded_nodes += 1

    return None, expanded_nodes, len(frontier)

def global_greedy(start_node: Node, heuristics=None):
    """ global greedy search considers the heuristic value of all nodes in the frontier """
    expanded_nodes = 0
    frontier = [start_node]
    explored = set()

    while frontier:
        current_node = min(frontier, key=lambda node: max(heuristic(node.state) for heuristic in heuristics))
        frontier.remove(current_node)

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        if current_node not in explored:
            explored.add(current_node)

            if current_node.is_dead_end():
                continue

            children = current_node.get_children()
            for child in children:
                if child not in explored and child not in frontier:
                    frontier.append(child)

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier) # No solution
