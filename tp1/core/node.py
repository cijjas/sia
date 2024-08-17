class Node:
    def __init__(self, state,parent=None, path_cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.heuristic = heuristic
        self.total_cost = self.path_cost + self.heuristic

    def __lt__(self, other):
        return self.total_cost < other.total_cost
    
    def get_path(self):
        node, path_back = self, []
        while node:
            path_back.append((node.action, node.state))
            node = node.parent
        return list(reversed(path_back))[1:]
        