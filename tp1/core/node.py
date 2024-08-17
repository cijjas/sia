class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action  # Track the action leading to this node
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.total_cost = self.path_cost + self.heuristic

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

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)
