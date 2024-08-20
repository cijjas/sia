import json
import csv
import sys
import os
import time
from core.heuristics import *
from core.utils.map_parser import parse_map
from core.algorithms.a_star import a_star
from core.algorithms.bfs import bfs
from core.algorithms.dfs import dfs
from core.algorithms.greedy import local_greedy, global_greedy
from core.algorithms.iddfs import iddfs
from core.models.state import State
from core.models.node import Node
from sokoban import parse_map

MAPS_PATH = "./maps"
OUTPUT_DIR = "output"
OUTPUT_FILE_NAME = "executions.csv"

# Map user input to the corresponding functions
algorithm_map = {
    "BFS": bfs,
    "DFS": dfs,
    "A_STAR": a_star,
    "GREEDY_LOCAL": local_greedy,
    "GREEDY_GLOBAL": global_greedy,
    "IDDFS": iddfs
}

# Map user input to the corresponding heuristic class
heuristics_map = {
    "MinManhattan": MinManhattan(),
    "MinEuclidean": MinEuclidean(),
    "DeadlockCorner" : DeadlockCorner(),
    "DeadlockWall" : DeadlockWall(),
    "MinManhattanBetweenPlayerAndBox" : MinManhattanBetweenPlayerAndBox(),
}

def main():
    if (len(sys.argv) != 2):
        print("Choose a map!")
        sys.exit(1)

    # Parse the JSON Config file
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

        maps = config['data_generation']['maps']
        algorithms = config['data_generation']['algorithms']
        iterations_for_average = config['data_generation']['iterations_for_average']
        heuristics = []
        for h in config['data_generation']['heuristics']:
            heuristics.append(heuristics_map[h])

    # Create Data Set : for each map and for each algorithm run X iterations and write them into a csv
    with open(f'{OUTPUT_DIR}/{OUTPUT_FILE_NAME}', mode='w', newline='') as file:
        writer = csv.writer(file)
        csv_header = ['map', 'algorithm', 'iteration', "execution_time", "expanded_nodes", "frontier_nodes", "success_or_failure", "cost", "solution_path"]
        writer.writerow(csv_header)
        for m in maps:
            map_data = parse_map(f'{MAPS_PATH}/{m}')
            initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])
            initial_node = Node(initial_state, None, None, 0)
            for a in algorithms:
                algorithm_function = algorithm_map[a]
                for i in range(iterations_for_average):
                    start_time = time.time()
                    if (a == "A_STAR" or a == "GREEDY_GLOBAL" or a == "GREEDY_LOCAL"):
                        solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node, heuristics)
                    elif (a == "BFS" or a == "DFS" or a == "IDDFS"):
                        solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node)
                    else:
                        print("Invalid Algorithm")
                        sys.exit(1)

                    execution_time = time.time() - start_time

                    row = [m.split('.')[0], a, i + 1, execution_time, expanded_nodes, frontier_nodes, "SUCCESS" if solution_path != None else "FAILURE", len(solution_path), solution_path]
                    writer.writerow(row)


    # Generate analytics


if __name__ == "__main__":
    main()