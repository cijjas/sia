import heapq
from collections import deque

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def manhattan_distance(state, goal):
    distance = 0
    for i in range(1, 9):  # 1 through 8
        x1, y1 = divmod(state.index(i), 3)
        x2, y2 = divmod(goal.index(i), 3)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def neighbors(state):
    neighbors = []
    index = state.index(0)  # Find the empty space
    x, y = divmod(index, 3)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for move in moves:
        new_x, new_y = x + move[0], y + move[1]
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_index = new_x * 3 + new_y
            new_state = list(state)
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            neighbors.append(tuple(new_state))
    return neighbors

def print_board(state):
    for i in range(0, 9, 3):
        print(state[i:i + 3])
    print()

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def a_star(start, goal):
    open_set = PriorityQueue()
    open_set.put(start, 0)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_distance(start, goal)}

    while not open_set.is_empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal)
                open_set.put(neighbor, f_score[neighbor])

    return None  # No solution found

def bfs(start, goal):
    open_set = deque([start])
    came_from = {}
    while open_set:
        current = open_set.popleft()
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in neighbors(current):
            if neighbor not in came_from:
                came_from[neighbor] = current
                open_set.append(neighbor)
    return None


def dfs(start, goal):
    open_set = [start]
    came_from = {}
    while open_set:
        current = open_set.pop()
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in reversed(neighbors(current)):
            if neighbor not in came_from:
                came_from[neighbor] = current
                open_set.append(neighbor)
    return None

def iddfs(start, goal):
    depth = 0
    while True:
        came_from = {}
        open_set = [start]
        while open_set:
            current = open_set.pop()
            if current == goal:
                return reconstruct_path(came_from, current)
            if len(came_from[current]) < depth:
                for neighbor in neighbors(current):
                    if neighbor not in came_from:
                        came_from[neighbor] = current
                        open_set.append(neighbor)
        depth += 1
    return None

# Example usage:
start_state = (2, 4, 6, 1, 3, 8, 0, 5, 7)  # Empty space is represented by 0
goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Goal configuration

solution = a_star(start_state, goal_state)

if solution:
    for i, state in enumerate(solution):
        print(f"Move {i + 1}")
        print_board(state)
else:
    print("No solution found")
