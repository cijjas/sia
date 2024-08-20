import json
import csv
import sys
import os
from tp1.core.structure.state import State
from tp1.core.structure.node import Node
from sokoban import parse_map
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.greedy import global_greedy
from algorithms.greedy import local_greedy
from algorithms.iddfs import iddfs
from algorithms.a_star import a_star

MAPS_PATH = "./maps"
OUTPUT_DIR = "output"
OUTPUT_FILE_NAME = "executions.csv"

def main():
    if (len(sys.argv) != 2):
        print("Choose a map!")
        sys.exit(1)

    # Parse the JSON Config file
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
        maps = config['data_generation']['maps']
        algorithms = config['data_generation']['algorithms']
        heuristics = config['data_generation']['heuristics']
        iterations_for_average = config['data_generation']['iterations_for_average']

    # Create Data Set : for each map and for each algorithm run X iterations and write them into a csv
    with open(f'{OUTPUT_DIR}/{OUTPUT_FILE_NAME}', mode='w', newline='') as file:
        writer = csv.writer(file)
        # These headers are missing : Execution Time, Success/Failure, Cost
        csv_header = ['map', 'algorithm', 'iteration', "expanded_nodes", "frontier_nodes", "solution_path"]
        writer.writerow(csv_header)
        for m in maps:
            map_data = parse_map(f'{MAPS_PATH}/{m}')
            initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'])
            initial_node = Node(initial_state, None, None, 0)
            for a in algorithms:
                for i in range(iterations_for_average):
                    if (a == "BFS"):
                        solution_path, expanded_nodes, frontier_nodes = bfs(initial_node)
                    elif (a == "DFS"):
                        solution_path, expanded_nodes, frontier_nodes = dfs(initial_node)
                    elif (a == "A_STAR"):
                        # Heuristics should be handled as an array in the a_star function
                        # max_heuristic_value functon should be computed in the algorithm function
                        # Matching between user input and the right heuristic in the application
                        print("A_STAR")
                    elif (a == "GREEDY_LOCAL"):
                        print("GREEDY_LOCAL")
                    elif (a == "GREEDY_GLOBAL"):
                        print("GREEDY_GLOBAL")
                    elif (a == "DFS"):
                        print("IDDFS")
                    else:
                        print("Invalid Algorithm")
                        sys.exit(1)

                    row = [m.split('.')[0], a, i, expanded_nodes, frontier_nodes, solution_path]
                    writer.writerow(row)


    # Generate analytics


if __name__ == "__main__":
    main()