from abc import ABC, abstractmethod
from core.heuristics import Heuristic
from core.models.node import Node
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple
from collections import deque
import heapq

class Algorithm(ABC):
    @abstractmethod
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        return self.search(draw_board)

class GlobalGreedy(Algorithm):
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        """ global greedy search considers the heuristic value of all nodes in the frontier """
        expanded_nodes = 0
        frontier = [start_node]
        explored = set()

        while frontier:
            current_node = min(frontier, key=lambda x: heuristic(x.state)) 
            frontier.remove(current_node)

            if current_node.is_goal():
                return current_node.get_path(), expanded_nodes, len(frontier)

            if current_node not in explored:
                expanded_nodes += 1
                explored.add(current_node)

                if current_node.is_dead_end():
                    continue

                children = current_node.get_children()
                frontier = children + frontier

        return None, expanded_nodes, len(frontier) # No solution

class BFS(Algorithm):
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        # Initialize Fr and Ex
        expanded_nodes = 0
        frontier = deque([start_node])
        explored = set()

        while frontier:
            current_node = frontier.popleft()
            
            if current_node.is_goal():
                return current_node.get_path(), expanded_nodes, len(frontier)
            
            expanded_nodes += 1
            for child in current_node.get_children():
                if child not in explored:
                    explored.add(child)
                    child.parent = current_node
                    frontier.append(child)

        return None, expanded_nodes, len(frontier) # No solution    

class DFS(Algorithm):
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        # Initialize Fr and Ex
        expanded_nodes = 0
        frontier = [start_node]  # stack
        explored = set()

        while frontier:
            current_node = frontier.pop()  
            
            if current_node.is_goal():
                return current_node.get_path(), expanded_nodes, len(frontier)
            
            expanded_nodes += 1
            explored.add(current_node)
            for child in current_node.get_children():
                if child not in explored:
                    child.parent = current_node
                    frontier.append(child)  # push stack

        return None, expanded_nodes, len(frontier) 

class LocalGreedy(Algorithm):
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        """ local greedy search considers the heuristic value of most recently expanded node's children """
        expanded_nodes = 0
        frontier = [start_node]
        explored = set()

        while frontier:
            current_node = frontier.pop(0)

            if current_node.is_goal():
                return current_node.get_path(), expanded_nodes, len(frontier)

            if current_node not in explored:
                expanded_nodes += 1
                explored.add(current_node)

                if current_node.is_dead_end():
                    continue

                children = current_node.get_children()
                children.sort(key=lambda x: heuristic(x.state))
                frontier = children + frontier

        return None, expanded_nodes, len(frontier) # No solution

class AStar(Algorithm):
    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        open_set = [] # expanded nodes
        expanded_nodes = 0
        g_score = {start_node: 0} # mapa de nodos a g_score
        f_score = {start_node: heuristic(start_node.state)}

        heapq.heappush(open_set, (f_score[start_node], start_node))

        while open_set:
            # heapq agarra por default el primer elemento de la tupla para comparar
            _, current_node = heapq.heappop(open_set) # Pop the smallest item off the heap, maintaining the heap invariant.
            
            expanded_nodes += 1
            if current_node.is_goal():
                return current_node.get_path(), expanded_nodes, len(open_set)

            for child in current_node.get_children():
                tentative_g_score = g_score[current_node] + 1 # asumiendo uniform cost
                if tentative_g_score < g_score.get(child, float('inf')): # si el nodo no esta en g_score, devuelve inf
                    g_score[child] = tentative_g_score
                    f_score[child] = tentative_g_score + heuristic(child.state)
                    heapq.heappush(open_set, (f_score[child], child))

        return None, expanded_nodes, len(open_set)

class IDDFS(Algorithm):
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
            found, remaining, child_expanded, child_frontier_count = IDDFS.dls(child, depth - 1, visited)
            expanded_nodes += child_expanded
            frontier_count += child_frontier_count
            if found:
                return (found, True, expanded_nodes, frontier_count)
            if remaining:
                any_remaining = True

        if any_remaining and depth == 1:
            frontier_count += 1
        return (None, any_remaining, expanded_nodes, frontier_count)

    def __call__(self, start_node: Node, heuristic: Optional[Heuristic] = lambda x: 0, draw_board: Optional[Callable[[Node], None] ] = None) -> Tuple[List[str], int, int]:
        depth = 0
        total_expanded_nodes = 0
        total_frontier_count = 0
        while True:
            visited = set()
            found, remaining, expanded_nodes, frontier_count = self.dls(start_node, depth, visited)
            total_expanded_nodes += expanded_nodes
            total_frontier_count += frontier_count
            if found:
                return found.get_path(), total_expanded_nodes, total_frontier_count
            if not remaining:
                return None, total_expanded_nodes, total_frontier_count
            depth += 1
