from core.state import State
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

class DeadlockCorner(Heuristic):
    """ Returns the sum of the euclidean distances between each box and its closest goal """
    def __call__(self, state: State) -> float:
        for box in state.boxes:
            if state.is_corner(box[0], box[1]) and box not in state.goals:
                return float('inf')
        return MinManhattan()(state)

