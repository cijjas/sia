from core.structure.node import Node
from core.structure.state import State
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
            expanded_nodes += 1
            explored.add(current_node)

            if current_node.is_dead_end():
                continue

            children = current_node.get_children()
            children.sort(key=lambda child: max(heuristic(child.state) for heuristic in heuristics))
            frontier = children + frontier

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
            expanded_nodes += 1
            explored.add(current_node)

            if current_node.is_dead_end():
                continue

            children = current_node.get_children()
            frontier = children + frontier

    return None, expanded_nodes, len(frontier) # No solution

