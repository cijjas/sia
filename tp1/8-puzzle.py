import heapq
import time

class State:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        if isinstance(other, State):
            return order_min_corner(self.state) == order_min_corner(other.state)
        return False

    def __lt__(self, other):
        if isinstance(other, State):
            return order_min_corner(self.state) < order_min_corner(other.state)
        return False

    def __hash__(self):
        return hash(order_min_corner(self.state))

    def __repr__(self):
        return str(self.state)

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def misplaced_tiles(state, goal):
    return sum(s != g for s, g in zip(state.state, goal.state))

def manhattan_distance(state, goal):
    distance = 0
    for i in range(1, 9):  # 1 through 8
        x1, y1 = divmod(state.state.index(i), 3)
        x2, y2 = divmod(goal.state.index(i), 3)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def neighbors(state):
    neighbors = []
    empty_index = state.state.index(0)
    empty_x, empty_y = divmod(empty_index, 3)

    for i, (dx, dy) in enumerate(((0, 1), (1, 0), (0, -1), (-1, 0))):
        new_x, new_y = empty_x + dx, empty_y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_index = new_x * 3 + new_y
            new_state = list(state.state)
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            neighbors.append(State(tuple(new_state)))

    return neighbors

def print_board(state):
    for i in range(0, 9, 3):
        print(state.state[i:i + 3])
    print()

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def a_star(start, goal):
    start = State(order_min_corner(start))
    goal = State(order_min_corner(goal))
    open_set = PriorityQueue()
    open_set.put(start, 0)
    explored = set()
    expanded_nodes = 0

    came_from = {}
    g_score = {start: 0}
    f_score = {start: misplaced_tiles(start, goal)}

    while not open_set.is_empty():
        current = open_set.get()

        if current == goal:
            print(f"Expanded nodes: {expanded_nodes}")
            return reconstruct_path(came_from, current)

        explored.add(current)
        expanded_nodes += 1

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1

            if neighbor not in explored and (neighbor not in g_score or tentative_g_score < g_score[neighbor]):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + misplaced_tiles(neighbor, goal)
                open_set.put(neighbor, f_score[neighbor])

    return None  # No solution found

def global_greedy(start, goal):
    start = State(order_min_corner(start))
    goal = State(order_min_corner(goal))
    frontier = []
    explored = set()
    came_from = {}
    expanded_nodes = 0

    heapq.heappush(frontier, (0, start))
    explored.add(start)
    came_from[start] = None

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            print(f"Expanded nodes: {expanded_nodes}")
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current):
            if neighbor not in explored:
                heapq.heappush(frontier, (misplaced_tiles(neighbor, goal), neighbor))
                explored.add(neighbor)
                came_from[neighbor] = current

        expanded_nodes += 1

    return None

def order_min_corner(state):
    """ finds the corner with the lowest value and rotates the board so that it is in the upper left corner """
    # 0 1 2     2 5 8       8 7 6       6 3 0
    # 3 4 5 ->  1 4 7 ->    5 4 3 ->    7 4 1
    # 6 7 8     0 3 6       2 1 0       8 5 2
    min_corner = min(state[0], state[2], state[6], state[8])
    if min_corner == state[0]:
        return state
    elif min_corner == state[2]:
        return (state[2], state[5], state[8], state[1], state[4], state[7], state[0], state[3], state[6])
    elif min_corner == state[6]:
        return (state[6], state[3], state[0], state[7], state[4], state[1], state[8], state[5], state[2])
    else:
        return (state[8], state[7], state[6], state[5], state[4], state[3], state[2], state[1], state[0])

def main():
    # Example usage:
    start_state = (8, 0, 6, 5, 4, 7, 2, 3, 1)  # Empty space is represented by 0
    goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Goal configuration

    time_array = []
    for k in range(1):
        start_time = time.time()

        solution = a_star(start_state, goal_state)

        end_time = time.time()
        time_array.append(end_time - start_time)

    if solution:
        for i, state in enumerate(solution):
            print(f"Move {i + 1}")
            if state:
                print_board(state)
            
    else:
        print("No solution found")
    
    print(f"Time taken: {sum(time_array) / len(time_array):.6f} seconds")
    print(f"Number of moves: {len(solution) - 1}")


if __name__ == "__main__":
    main()