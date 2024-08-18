

# Depth-limited search
def dls(node, depth):
    expanded_nodes = 0  
    if depth == 0:
        expanded_nodes += 1  
        return (node, True, expanded_nodes) if node.is_goal() else (None, True, expanded_nodes)
    elif depth > 0:
        any_remaining = False
        for child in node.get_children():
            found, remaining, child_expanded = dls(child, depth - 1)
            expanded_nodes += child_expanded 
            if found is not None:
                return (found, True, expanded_nodes)
            if remaining:
                any_remaining = True
        return (None, any_remaining, expanded_nodes)


# iterative deepening dfs
def iddfs(root):
    depth = 0
    total_expanded_nodes = 0  
    while True:
        print(f"Depth: {depth}")
        found, remaining, expanded_nodes = dls(root, depth)
        total_expanded_nodes += expanded_nodes  
        if found is not None:
            return found.get_path(), total_expanded_nodes 
        if not remaining:
            return None, total_expanded_nodes 
        depth += 1