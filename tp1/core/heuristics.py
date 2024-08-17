def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def min_manhattan(target_positions: set):
    def heuristic(state):
        total_distance = 0
        for box in state.box_positions:
            min_distance = float('inf')
            for target in target_positions:
                distance = manhattan(box, target)
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
                distance = euclidean(box, target)
                if distance < min_distance:
                    min_distance = distance
            total_distance += min_distance
        return total_distance
    return heuristic



