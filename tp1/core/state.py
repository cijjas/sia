class State:
    def __init__(self, walls: set, goals: set, boxes: set, player):
        self.walls = walls
        self.goals = goals
        self.boxes = boxes
        self.player = player

    def retrieve(self, x, y) -> str:
        if (x, y) in self.walls: # O(1)
            return 'wall'
        elif (x, y) in self.goals:
            return 'goal'
        elif (x, y) in self.boxes:
            return 'box'
        elif (x, y) == self.player:
            return 'player'
        else:
            return None

    def is_in_corner(self, x, y):
        north = self.retrieve(x, y - 1)
        south = self.retrieve(x, y + 1)
        east = self.retrieve(x + 1, y)
        west = self.retrieve(x - 1, y)

        # we define a a circular list to check the corners
        neighbors = [north, east, south, west]

        # if there are two walls in a row, we have a corner
        for i in range(4):
            if neighbors[i%4] == 'wall' and neighbors[(i + 1)%4] == 'wall':
                return True

    def __eq__(self, other):
        return self.boxes == other.boxes and self.player == other.player

    def __hash__(self):
        return hash((tuple(self.boxes), self.player))

    def __str__(self):
        objects = {
            'walls': self.walls,
            'goals': self.goals,
            'boxes': self.boxes,
            'player': self.player
        }
        return str(objects)

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
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if self.can_move(dx, dy):
                actions.append((dx, dy))
        return actions

    def __str__(self):
        return f'State (\'{self.boxes}\', {self.player})'
