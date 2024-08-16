def manhattan_distance(state):
    min_distance = float('inf')
    px, py = state.player_position
    for gy, row in enumerate(state.goals):
        for gx, goal in enumerate(row):
            if goal:  
                distance = abs(px - gx) + abs(py - gy)
                if distance < min_distance:
                    min_distance = distance
    return min_distance
