import functions as f

def min_manhattan(target_positions: set):
    def heuristic(state):
        total_distance = 0
        for box in state.box_positions:
            min_distance = float('inf')
            for target in target_positions:
                distance = f.manhattan(box, target)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance
    return heuristic

def min_euclidean(target_positions: set):
    def heuristic(state):
        total_distance = 0
        for box in state.box_positions:
            min_distance = float('inf')
            for target in target_positions:
                distance = f.euclidean(box, target)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance
    return heuristic



