from core.models.state import State
from core.models.node import Node
from abc import ABC, abstractmethod

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def heuristic_combinator(node, heuristics):
    return max(heuristic(node.state) for heuristic in heuristics)

class Heuristic(ABC):
    @abstractmethod
    def __call__(self, state: State) -> float:
        pass

    def __str__(self):
        return self.__class__.__name__

# Calculates the sum of the manhattan distances between the boxes and their closest target
class ManhattanDistance1(Heuristic):
    def __call__(self, state: State) -> float:
        total_distance = 0
        for box in state.boxes:
            min_for_box = float('inf')
            for goal in state.goals:
                box_to_goal_distance = manhattan(box, goal)
                if box_to_goal_distance < min_for_box:
                    min_for_box = box_to_goal_distance
            total_distance += min_for_box
        return total_distance

# Calculates the sum of the manhattan distances to the closest box and the distances from each box to their closest target
class ManhattanDistance2(Heuristic):
    def __call__(self, state: State) -> float:
        total_distance = 0
        min_for_a_box = float('inf')

        for box in state.boxes:
            player_to_box_distance = manhattan(box, state.player)
            if player_to_box_distance < min_for_a_box:
                min_for_a_box = player_to_box_distance

        for box in state.boxes:
            min_for_box = float('inf')
            for goal in state.goals:
                box_to_goal_distance = manhattan(box, goal)
                if box_to_goal_distance < min_for_box:
                    min_for_box = box_to_goal_distance
            total_distance += min_for_box

        return total_distance + min_for_a_box

def wall_and_path_penalty( state, box):
    penalty = 0
    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]: 
        adjacent = (box[0] + direction[0], box[1] + direction[1])
        if adjacent in state.walls:
            penalty += 2  
    return penalty

# igual que m2 pero con un poco de inteligencia
class Smarthattan(Heuristic):
    def __call__(self, state: State) -> float:
        total_distance = 0
        min_player_to_box = float('inf')

        for box in state.boxes:
            player_to_box_distance = manhattan(box, state.player)
            if player_to_box_distance < min_player_to_box:
                min_player_to_box = player_to_box_distance

        for box in state.boxes:
            min_box_to_goal = float('inf')
            for goal in state.goals:
                box_to_goal_distance = manhattan(box, goal)

                # Agregarle un changüi si están en la misma fila o columna
                if (goal[0] == box[0] and state.player[0] == box[0]) or (goal[1] == box[1] and state.player[1] == box[1]):
                    box_to_goal_distance -= 1  

                if box_to_goal_distance < min_box_to_goal:
                    min_box_to_goal = box_to_goal_distance

            total_distance += min_box_to_goal

        # penalizar si se quiere mover la caja contra una pared
        total_distance += wall_and_path_penalty(state, box)
        return total_distance + min_player_to_box

# Calculates the sum of the manhattan distance to the closest box, and from the box the closest distance to a target
class ManhattanDistance3(Heuristic):
    def __call__(self, state: State) -> float:
        min_box, min_for_a_box = (None, float('inf'))

        for box in state.boxes:
            player_to_box_distance = manhattan(box, state.player)
            if player_to_box_distance < min_for_a_box:
                min_box, min_for_a_box = box, player_to_box_distance

        min_for_target = float('inf')
        for goal in state.goals:
            box_to_goal_distance = manhattan(min_box, goal)
            if box_to_goal_distance < min_for_target:
                min_for_target = box_to_goal_distance

        return min_for_target + min_for_a_box

# Calculates the sum of the euclidean distances between the boxes and their closest target
class EuclideanDistance(Heuristic):
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

# Calculates whether the state is a deadlock or not
class Deadlock(Heuristic):
    """ Heuristic that returns infinity if the state is a deadlock, 0 otherwise """
    def __call__(self, state: State) -> float:
        if state.is_deadlock():
            return float('inf')
        return 0

# We can say that given a wall with no goals or holes, it is a deadlock if a box is next to it
class DeadlockCorner(Heuristic):
    def __call__(self, state: State) -> float:
        if state.is_deadlock_corner():
            return float('inf')
        return 0


class BasicHeuristic(Heuristic):
    def __call__(self, state: State) -> float:
        #  returns the amount of boxes that are not in the correct position
        return len([box for box in state.boxes if box not in state.goals])