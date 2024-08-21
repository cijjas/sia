from core.models.state import State
from abc import ABC, abstractmethod

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

class Heuristic(ABC):
    @abstractmethod
    def __call__(self, state: State) -> float:
        pass

    def __str__(self):
        return self.__class__.__name__

class MinManhattan2(Heuristic):
    """ Returns the sum of the manhattan distances between each box and its closest goal """
    def __call__(self, state: State) -> float:
        total_distance = 0
        min_for_a_box = float('inf')

        # Calculate the minimum distance from the player to any box
        for box in state.boxes:
            player_to_box_distance = abs(box[0] - state.player[0]) + abs(box[1] - state.player[1])
            if player_to_box_distance < min_for_a_box:
                min_for_a_box = player_to_box_distance

        # Calculate the minimum distance from each box to the nearest goal
        for box in state.boxes:
            min_for_box = float('inf')
            for goal in state.goals:
                box_to_goal_distance = abs(box[0] - goal[0]) + abs(box[1] - goal[1])
                if box_to_goal_distance < min_for_box:
                    min_for_box = box_to_goal_distance
            total_distance += min_for_box

        # Add the minimum player-to-box distance to the total distance
        return total_distance + min_for_a_box

class MinManhattan(Heuristic):
    """ Returns the sum of the manhattan distances between each box and its closest goal """
    def __call__(self, state: State) -> float:
        total_distance = 0
        for box in state.boxes:
            min_distance = float('inf')
            for goal in state.goals:
                distance = manhattan(box, goal)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance

class MinEuclidean(Heuristic):
    """ Returns the sum of the euclidean distances between each box and its closest goal """
    def __call__(self, state: State) -> float:
        total_distance = 0
        for box in state.boxes:
            min_distance = float('inf')
            for goal in state.goals:
                distance = euclidean(box, goal)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance

class DeadlockCorner(Heuristic):
    """ Returns the sum of the euclidean distances between each box and its closest goal """
    def __call__(self, state: State) -> float:
        if state.is_deadlock():
            return float('inf')
        return MinManhattan()(state)

# We can say that given a wall with no goals or holes, it is a deadlock if a box is next to it
class DeadlockWall(Heuristic):

    def __call__(self, state: State) -> float:
        # if a box is next to a wall
        if not state.any_box_next_to_wall():
            return MinManhattan()(state)

        for box, wall in state.get_boxes_next_to_wall():
            if state.no_salvation_from_wall(box, wall):
                return float('inf')
        return MinManhattan()(state)

class MinManhattanBetweenPlayerAndBox(Heuristic):
    """ Returns the sum of the manhattan distances between each player and its closest box """
    def __call__(self, state: State) -> float:
        total_distance = 0
        for box in state.boxes:
            min_distance = float('inf')
            for player in state.player:
                distance = manhattan(box, player)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance
