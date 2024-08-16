from core.node import Node

def dfs(initial_state):
    frontier = [Node(initial_state)]
    explored = set()

    while frontier:
        node = frontier.pop()

        if node.state.is_goal():
            return node.path() 
        
        explored.add(node.state)

        for child_state in node.state.generate_children():
            if child_state not in explored:
                frontier.append(Node(child_state, node))

    return None 
