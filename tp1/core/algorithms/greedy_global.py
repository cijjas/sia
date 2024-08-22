from core.models.node import Node
from core.models.state import State
from core.heuristics import Heuristic
from core.heuristics import heuristic_combinator
import heapq

# Similar to Local Greedy but the order is kept amongst all frontier nodes
# Also similar A* but path cost is not taken into account
def greedy_global(start_node: Node, heuristics=None):
    frontier = []
    explored = set()
    expanded_nodes = 0

    heapq.heappush(frontier, (0, start_node))
    explored.add(start_node)

    while frontier:
        _, current_node = heapq.heappop(frontier)

        if current_node.is_goal():
            return current_node.get_path(), expanded_nodes, len(frontier)

        children = current_node.get_children()
        for child in children:
            if child not in explored:
                current_h_score = heuristic_combinator(child, heuristics)
                heapq.heappush(frontier, (current_h_score, child))
                explored.add(child)

        expanded_nodes += 1

    return None, expanded_nodes, len(frontier) # No solution
