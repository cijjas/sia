import queue
from core.node import Node
def bfs_solve(start_state):
    initial_node = Node(start_state)
    frontier = queue.Queue()
    frontier.put(initial_node)
    seen = set([start_state])

    while not frontier.empty():
        current_node = frontier.get()

        if current_node.is_goal():
            print("Goal reached!")
            return current_node.path
        
        for successor in current_node.generate_successors():
            if successor.state not in seen:
                seen.add(successor.state)
                frontier.put(successor)

    print("No solution found.")
    return None
