from typing import List, Tuple

class State:
    def __init__(self, walls: set, goals: set, boxes: set, player: tuple, spaces: set, deadlock_areas: set = set()):
        self.walls = walls
        self.goals = goals
        self.boxes = boxes
        self.player = player
        self.spaces = spaces
        self._hash_value = None
        self.deadlock_areas: set = deadlock_areas


    def __eq__(self, other):
        return self.boxes == other.boxes and self.player == other.player

    def __hash__(self):
        if self._hash_value is None:
            self._hash_value = hash((tuple(self.boxes), self.player))
        return self._hash_value

    def init_deadlock_areas(self):
        corner_deadlock_areas: set = self._init_corner_deadlock_areas(self.walls)
        wall_deadlock_areas: set = self._init_wall_deadlock_areas(corner_deadlock_areas)
        self.deadlock_areas = corner_deadlock_areas.union(wall_deadlock_areas)

    def _init_corner_deadlock_areas(self, walls: set) -> set:
        """ Initializes the corner deadlock areas set"""
        corner_deadlock_areas = set()
        for wall in walls:
            x, y = wall
            if (x, y-1) in self.walls and (x-1, y) in self.walls and (x-1, y-1) in self.spaces:
                corner_deadlock_areas.add((x-1, y-1))
            if (x, y-1) in self.walls and (x+1, y) in self.walls and (x+1, y-1) in self.spaces:
                corner_deadlock_areas.add((x+1, y-1))
            if (x, y+1) in self.walls and (x-1, y) in self.walls and (x-1, y+1) in self.spaces:
                corner_deadlock_areas.add((x-1, y+1))
            if (x, y+1) in self.walls and (x+1, y) in self.walls and (x+1, y+1) in self.spaces:
                corner_deadlock_areas.add((x+1, y+1))
        return corner_deadlock_areas

    # Looks for situations like these:
    #               ##
    #   ########    #
    #   #      #    #
    #               #
    #               ##
    def _init_wall_deadlock_areas(self, corner_deadlock_areas: set) -> set:
        """ Initializes the wall deadlock areas set """
        deadlocks = set()

        # Iterate through each pair of corner deadlocks
        for (x1, y1) in corner_deadlock_areas:
            for (x2, y2) in corner_deadlock_areas:
                if (x1, y1) != (x2, y2):
                    new_deadlock_line: set = set()
                    diff = 0
                    # Check if they share an x coordinate (vertical wall)
                    if x1 == x2:
                        diff = abs(y1 - y2)
                        for y in range(min(y1, y2) + 1, max(y1, y2)):
                            possible_spaces: set = self.spaces.union(self.boxes)
                            possible_spaces.add(self.player)
                            possible_spaces = possible_spaces.difference(self.goals)
                            if (x1, y) in possible_spaces and ( (x1-1, y) in self.walls or (x1+1, y) in self.walls):
                                new_deadlock_line.add((x1, y))
                    # Check if they share a y coordinate (horizontal wall)
                    elif y1 == y2:
                        diff = abs(x1 - x2)
                        for x in range(min(x1, x2) + 1, max(x1, x2)):
                            possible_spaces: set = self.spaces.union(self.boxes)
                            possible_spaces.add(self.player)
                            possible_spaces = possible_spaces.difference(self.goals)
                            if (x, y1) in possible_spaces and ( (x, y1-1) in self.walls or (x, y1+1) in self.walls):
                                new_deadlock_line.add((x, y1))

                    if len(new_deadlock_line) == diff-1 and len(new_deadlock_line) > 0:
                        deadlocks = deadlocks.union(new_deadlock_line)
                    
        return deadlocks

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
        return State(self.walls, self.goals, new_boxes, new_player, self.spaces, self.deadlock_areas)

    def is_goal(self):
        return all(box in self.goals for box in self.boxes)

    def get_actions(self):
        actions = []
        states = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if self.can_move(dx, dy) :
                states.append(self.move_player(dx, dy))
                actions.append((dx, dy))
        return actions, states

    def any_box_next_to_wall(self) -> bool:
        for box in self.boxes:
            if self.box_next_to_wall(box) is not None:
                return True
        return False

    def box_next_to_wall(self, box) -> tuple[int, int]:
        """ Returns the wall location. None if there is no wall next to the box """
        x, y = box
        if (x - 1, y) in self.walls:
            return (x - 1, y)
        if (x + 1, y) in self.walls:
            return (x + 1, y)
        if (x, y - 1) in self.walls:
            return (x, y - 1)
        if (x, y + 1) in self.walls:
            return (x, y + 1)
        return None

    def get_boxes_next_to_wall(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """ Returns a tuple with the box and the wall next to it. None if there is no box next to a wall """
        boxes = []
        for box in self.boxes:
            wall = self.box_next_to_wall(box)
            if wall is not None:
                boxes.append((box, wall))
        return boxes


    def check_goals_along_wall(self, x, y, dir_parallel_to_wall):
        """Helper function to check for goals along the wall in a given direction."""
        while self.can_move(x, y):
            if (x, y) in self.goals:
                return True
            x += dir_parallel_to_wall[0]
            y += dir_parallel_to_wall[1]
        return False

    def check_walls_along_wall(self, wall_x, wall_y, dir_parallel_to_wall, dir_perpendicular_to_wall):
        """Helper function to check for walls along the wall in a given direction."""
        while (wall_x, wall_y) in self.walls:
            wall_x += dir_parallel_to_wall[0]
            wall_y += dir_parallel_to_wall[1]
        dir_x, dir_y = wall_x, wall_y
        dir_x -= dir_perpendicular_to_wall[0]
        dir_y -= dir_perpendicular_to_wall[1]
        return (dir_x, dir_y) in self.walls

    def __str__(self):
        return f'State (\'{self.boxes}\', {self.player})'

    def is_deadlock(self):
        """ Returns True if the state is a deadlock by checking the deadlock areas """
        for box in self.boxes:
            if box in self.deadlock_areas and box not in self.goals:
                return True
        return False

    