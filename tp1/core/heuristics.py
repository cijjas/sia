from core.structure.state import State
from abc import ABC, abstractmethod

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

class Heuristic(ABC):
    @abstractmethod
    def __call__(self, state: State) -> float:
        pass

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

# Careful with tunnels to do check
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
    
