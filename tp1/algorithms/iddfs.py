from core.node import Node

# basado en https://academy.finxter.com/python-iterative-deepening-depth-first-search-dfs-algorithm/
def dls(node: Node, depth, visited):
    expanded_nodes = 0
    frontier_count = 0
    if node.state in visited:
        return (None, False, expanded_nodes, frontier_count)
    
    visited.add(node.state)
    if depth == 0:
        expanded_nodes += 1
        is_frontier = len(node.get_children()) > 0 if not node.is_goal() else False
        frontier_count += 1 if is_frontier else 0
        return (node, True, expanded_nodes, frontier_count) if node.is_goal() else (None, True, expanded_nodes, frontier_count)
    
    any_remaining = False
    for child in node.get_children():
        found, remaining, child_expanded, child_frontier_count = dls(child, depth - 1, visited)
        expanded_nodes += child_expanded
        frontier_count += child_frontier_count
        if found:
            return (found, True, expanded_nodes, frontier_count)
        if remaining:
            any_remaining = True

    if any_remaining and depth == 1:
        frontier_count += 1
    return (None, any_remaining, expanded_nodes, frontier_count)

def iddfs(root: Node):
    depth = 0
    total_expanded_nodes = 0
    total_frontier_count = 0
    while True:
        visited = set()
        found, remaining, expanded_nodes, frontier_count = dls(root, depth, visited)
        total_expanded_nodes += expanded_nodes
        total_frontier_count += frontier_count
        if found:
            return found.get_path(), total_expanded_nodes, total_frontier_count
        if not remaining:
            return None, total_expanded_nodes, total_frontier_count
        depth += 1
