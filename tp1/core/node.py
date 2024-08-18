from core.state import State
from core.heuristics import Heuristic

class Node:
    def __init__(self, state: State, parent=None, action=None, path_cost=0, heuristic: Heuristic = lambda x: 0):
        self.state = state
        self.parent = parent
        self.action = action  # Track the action leading to this node
        self.action = action  # Track the action leading to this node
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.heuristic_value = self.heuristic(state)
        self.total_cost = self.path_cost + self.heuristic_value

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def get_path(self):
        actions = []
        node = self
        while node.parent:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    def __str__(self):
        return str(self.state)

    def get_children(self):
        actions, states = self.state.get_actions()
        return [Node(state, self, action, self.path_cost + 1, self.heuristic) for action, state in zip(actions, states)]

    def is_goal(self):
        return self.state.is_goal()

    def is_dead_end(self):
        return self.state.is_deadlock()

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)
