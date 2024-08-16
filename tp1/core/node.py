class Node:
    def __init__(self, state, parent=None, action=None, path=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.path = path or []
        
    def generate_successors(self):
        successors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible directions to move
        for direction in directions:
            if self.state.can_move(*direction):
                new_state = self.state.move_player(*direction)
                if new_state:
                    new_path = self.path + [direction]
                    successors.append(Node(new_state, self, direction, new_path))
        return successors

    def is_goal(self):
        return self.state.goal_reached()
