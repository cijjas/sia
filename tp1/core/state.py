class State:
    def __init__(self, board):
        self.board = board
        self.player_position = self.find_player()
        # now for the clean board we need to make a copy of the board
        # and replace the player with a space
        self.clean_board = [list(row) for row in board]
        x, y = self.player_position
        self.clean_board[y][x] = ' '
        # we also need to replace the boxes with spaces
        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                if char == '$':
                    self.clean_board[y][x] = ' '

    def find_player(self):
        for y, row in enumerate(self.board):
            for x, char in enumerate(row):
                if char == '@':
                    return (x, y)

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
            print('Pushing box')
            self.board[target2[1]][target2[0]] = '$'  

        self.board[y][x] = self.clean_board[y][x]
        self.board[target[1]][target[0]] = '@'
        self.player_position = target

