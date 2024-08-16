class State:
    def __init__(self, board, player_position=None, goals=None):
        self.board = tuple(tuple(row) for row in board)  # Immutable board
        self.player_position = player_position or self.find_player(board)
        self.goals = goals if goals is not None else self.extract_goals(board)

    def __hash__(self):
        return hash((self.board, self.player_position))

    def __eq__(self, other):
        return isinstance(other, State) and self.board == other.board and self.player_position == other.player_position

    def find_player(self, board):
        for y, row in enumerate(board):
            for x, char in enumerate(row):
                if char == '@':
                    return (x, y)

    def extract_goals(self, board):
        return tuple(tuple(char == '.' for char in row) for row in board)

    def can_move(self, dx, dy):
        x, y = self.player_position
        target = (x + dx, y + dy)
        target2 = (x + 2 * dx, y + 2 * dy)
        if self.board[target[1]][target[0]] == '#' or (self.board[target[1]][target[0]] in '$*' and self.board[target2[1]][target2[0]] in '#$*'):
            return False
        return True

    def move_player(self, dx, dy):
        if not self.can_move(dx, dy):
            return None
        new_board = list(list(row) for row in self.board)  # Temporarily convert to lists to manipulate
        new_state = State(new_board, self.player_position, self.goals)  # Recreate with the same constructor
        # Proceed with move logic
        x, y = new_state.player_position
        target = (x + dx, y + dy)
        target2 = (x + 2 * dx, y + 2 * dy)
        if new_board[target[1]][target[0]] in '$*':
            new_box_symbol = '*' if new_state.goals[target2[1]][target2[0]] else '$'
            new_board[target2[1]][target2[0]] = new_box_symbol
            new_board[target[1]][target[0]] = '.' if new_state.goals[target[1]][target[0]] else ' '
        new_board[y][x] = '.' if new_state.goals[y][x] else ' '
        new_board[target[1]][target[0]] = '@'
        new_state.player_position = target
        new_state.board = tuple(tuple(row) for row in new_board)  # Convert back to immutable structure
        return new_state

    def goal_reached(self):
        return all(char != '$' for row in self.board for char in row)
