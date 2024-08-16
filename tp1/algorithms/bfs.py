from collections import deque
from core.node import Node

def bfs(initial_state):
    frontier = deque([Node(initial_state)])
    explored = set()

    while frontier:
        node = frontier.popleft()

        if node.state.is_goal():
            return node.path()  

        explored.add(node.state)

        for child_state in node.state.generate_children():
            if child_state not in explored:
                frontier.append(Node(child_state, node))

    return None 
