from abc import ABC, abstractmethod
from core.state import State

class Algorithm(ABC):
    @abstractmethod
    def get(self) -> State:
        pass

    @abstractmethod
    def put(self, state: State) -> None:
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

    def get(self) -> State:
        return self.frontier.pop()

    def put(self, state: State) -> None:
        self.frontier.append(state)

    def is_empty(self) -> bool:
        return len(self.frontier) == 0

    def get_frontier(self) -> list:
        return self.frontier

class BFS(Algorithm):
    def __init__(self):
        self.frontier = []

    def get(self) -> State:
        return self.frontier.pop(0)

    def put(self, state: State) -> None:
        self.frontier.append(state)

    def is_empty(self) -> bool:
        return len(self.frontier) == 0

    def get_frontier(self) -> list:
        return self.frontier

