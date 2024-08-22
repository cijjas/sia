from core.models.node import Node
from core.models.state import State
from core.heuristics import Heuristic
from core.heuristics import heuristic_combinator
from collections import deque

# Similar to a DFS but the decision on which to expand next is based on a Heuristic instead of having position preference
def greedy_local(start_node: Node, heuristics=None):
    frontier = deque()
    explored = set()
    expanded_nodes = 0

    frontier.append(start_node)
    explored.add(start_node)

    while frontier:
        current_node = frontier.pop()

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        # Sorting before insertion, so the order is amongst childrens and not globally
        children = current_node.get_children()
        children.sort(key=lambda child: heuristic_combinator(child, heuristics), reverse = True)
        for child in children:
            if child not in explored:
                frontier.append(child)
                explored.add(child)

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier)
