class State:
    def __init__(self, walls, goals, boxes, player):
        self.walls = walls
        self.goals = goals
        self.boxes = boxes
        self.player = player


    def __eq__(self, other):
        return self.boxes == other.boxes and self.player == other.player
    
    def __hash__(self):
        return hash((tuple(self.boxes), self.player))
    
    def __str__(self):
        objects = {
            'walls': self.walls,
            'goals': self.goals,
            'boxes': self.boxes,
            'player': self.player,
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
            box_index = new_boxes.index(new_player)
            new_box = (new_player[0] + dx, new_player[1] + dy)
            new_boxes[box_index] = new_box
        return State(self.walls, self.goals, new_boxes, new_player)
    
    def is_goal(self):
        return all(box in self.goals for box in self.boxes)
    
    
       