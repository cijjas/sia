from core.algorithms import Algorithm
from core.state import State
from typing import Optional
from time import sleep

def get_children(state: State) -> list[State]:
    children = []
    for action in state.get_actions():
        new_state = state.move_player(*action)
        if new_state is not None:
            children.append(new_state)
    return children

def search(algorithm: Algorithm, state: State, draw_board: Optional[callable] = None) -> Optional[State]:
    explored = set()
    algorithm.put(state)

    while not algorithm.is_empty():

        current = algorithm.get()

        if current.is_goal():
            return current

        if current not in explored:
            explored.add(current)

            for child in get_children(current):
                if child not in explored:
                    algorithm.put(child)

        if draw_board is not None:
            draw_board(current)

        #sleep(0.1)

    return None
