from collections import deque
from core.node import Node

class BFS:
    def __init__(self, initial_state):
        self.initial_state = initial_state

    def solve(self):
        # Initial Node
        root_node = Node(state=self.initial_state)
        
        if root_node.state.is_goal():
            return root_node.get_path()  # Return the path if initial state is goal
        
        # Frontier is a queue with the initial node
        frontier = deque([root_node])
        # Explored is a set to track visited states
        explored = set()
        
        while frontier:
            current_node = frontier.popleft()
            explored.add(current_node.state)
            
            for action, next_state in current_node.state.get_successors():
                if next_state not in explored and not any(node.state == next_state for node in frontier):
                    child_node = Node(state=next_state, parent=current_node, action=action)
                    if child_node.state.is_goal():
                        return child_node.get_path()  # Return the path once goal is found
                    frontier.append(child_node)
                    
        return None  # If no solution is found
