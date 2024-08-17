from collections import deque
from core.node import Node

def bfs(start_node):
    # Initialize Fr and Ex
    frontier = deque([start_node])
    explored = set()

    # Check if start node is goal
    if start_node.state.is_goal():
        return []

    while frontier:
        current_node = frontier.popleft()
        explored.add(current_node.state)

        for action in current_node.state.get_actions():
            next_state = current_node.state.move_player(*action)
            if next_state and next_state not in explored:
                next_node = Node(next_state, current_node, action, current_node.path_cost + 1)
                if next_node.state.is_goal():
                    return next_node.get_path()
                frontier.append(next_node)
    return None
