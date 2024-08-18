class State:
    def __init__(self, walls: set, goals: set, boxes: set, player):
        self.walls = walls
        self.goals = goals
        self.boxes = boxes
        self.player = player
        self._hash_value = None


    def __eq__(self, other):
        return self.boxes == other.boxes and self.player == other.player

    def __hash__(self):
        if self._hash_value is None:
            self._hash_value = hash((tuple(self.boxes), self.player))
        return self._hash_value

    

    def print_state(self):
        for y in range(0, 6):
            for x in range(0, 6):
                if (x, y) in self.walls:
                    print('#', end='')
                elif (x, y) in self.goals:
                    print('.', end='')
                elif (x, y) in self.boxes:
                    print('$', end='')
                elif (x, y) == self.player:
                    print('@', end='')
                else:
                    print(' ', end='')
            print()


    def can_move(self, dx, dy):
        x, y = self.player
        new_x, new_y = x + dx, y + dy
        if (new_x, new_y) in self.walls:
            return False
        if (new_x, new_y) in self.boxes:
            new_box_x, new_box_y = new_x + dx, new_y + dy
            if (new_box_x, new_box_y) in self.walls or (new_box_x, new_box_y) in self.boxes:
                return False
        return True

    def move_player(self, dx, dy):
        if not self.can_move(dx, dy):
            return None
        new_player = (self.player[0] + dx, self.player[1] + dy)
        new_boxes = self.boxes.copy()
        if new_player in new_boxes:
            new_boxes.remove(new_player)
            new_box = (new_player[0] + dx, new_player[1] + dy)
            new_boxes.add(new_box)
        return State(self.walls, self.goals, new_boxes, new_player)

    def is_goal(self):
        return all(box in self.goals for box in self.boxes)

    def get_actions(self):
        actions = []
        states = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if self.can_move(dx, dy) and not self.is_deadlock():
                states.append(self.move_player(dx, dy))
                actions.append((dx, dy))
        return actions, states

    def __str__(self):
        return f'State (\'{self.boxes}\', {self.player})'
    
    def is_deadlock(self):
        return self._corner_deadlock()

    def _corner_deadlock(self):
        
        for box in self.boxes:
            if box not in self.goals:
                x, y = box
                if ((x - 1, y) in self.walls or (x + 1, y) in self.walls) and \
                   ((x, y - 1) in self.walls or (x, y + 1) in self.walls):
                    return True
        return False
    