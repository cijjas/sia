from core.action import Action
class State:
    def __init__(self, board):
        self.board = board
        self.player_position = self.find_player()
        self.goal_positions = self.find_goals()
        self.box_actions = []
        self.clean_board = [list(row) for row in board]
        x, y = self.player_position
        self.clean_board[y][x] = ' '
        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                if char == '$':
                    self.clean_board[y][x] = ' '    

    def find_player(self):
        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                if char == '@':
                    return (x, y)

    def find_goals(self):
        goals = []
        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                if char == '.':
                    goals.append((x, y))
        return goals

    def move_player(self, dx, dy):
        x, y = self.player_position
        print(f'Moving player from ({x}, {y}) to ({x+dx}, {y+dy})')
        target = (x + dx, y + dy)
        target2 = (x + 2*dx, y + 2*dy)
        if self.board[target[1]][target[0]] == '#':
            print('Invalid move found #')
            return  
        elif self.board[target[1]][target[0]] == '$':
            if self.board[target2[1]][target2[0]] in '#$':
                print('Invalid move # or $ for target2')
                return
            print('Pushing box from ({}, {}) to ({}, {})'.format(target[0], target[1], target2[0], target2[1]))
            self.box_actions.append(Action('push', target, target2, self.player_position))
            self.board[target2[1]][target2[0]] = '$'

        self.board[y][x] = self.clean_board[y][x]
        self.board[target[1]][target[0]] = '@'
        self.player_position = target

    def undo_box_action(self):
        if len(self.box_actions) == 0:
            return
        action = self.box_actions.pop()
        print(f'Undoing box action from {action.target2} to {action.target1}')
        
        # Move player to the previous position
        x, y = self.player_position
        self.board[y][x] = self.clean_board[y][x]
        self.player_position = action.player_position
        self.board[self.player_position[1]][self.player_position[0]] = '@'
        print(f'Player position is now {self.player_position}')
        
        # Place the box in the correct position
        self.board[action.target1[1]][action.target1[0]] = '$'
        self.board[action.target2[1]][action.target2[0]] = self.clean_board[action.target2[1]][action.target2[0]]

    def is_solved(self):
        for x, y in self.goal_positions:
            if self.board[y][x] != '$':
                return False
        return True
