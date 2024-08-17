from abc import ABC, abstractmethod
from core.heuristics import Heuristic
from core.node import Node

class Algorithm(ABC):
    @abstractmethod
    def get(self) -> Node:
        pass

    @abstractmethod
    def put(self, state: Node) -> None:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def get_frontier(self) -> list:
        pass


class DFS(Algorithm):
    def __init__(self):
        self.frontier = []

    def get(self) -> Node:
        return self.frontier.pop()

    def put(self, node: Node) -> None:
        self.frontier.append(node)

    def is_empty(self) -> bool:
        return len(self.frontier) == 0

    def get_frontier(self) -> list:
        return self.frontier

class BFS(Algorithm):
    def __init__(self):
        self.frontier = []

    def get(self) -> Node:
        return self.frontier.pop(0)

    def put(self, state: Node) -> None:
        self.frontier.append(state)

    def is_empty(self) -> bool:
        return len(self.frontier) == 0

    def get_frontier(self) -> list:
        return self.frontier

class Greedy(Algorithm):
    def __init__(self, heuristic: Heuristic):
        self.frontier = []
        self.heuristic = heuristic

    def get(self) -> Node:
        self.frontier.sort(key=lambda state: self.heuristic(state))
        return self.frontier.pop(0)

    def put(self, node: Node) -> None:
        self.frontier.append(node)

    def is_empty(self) -> bool:
        return len(self.frontier) == 0

    def get_frontier(self) -> list:
        return self.frontier
