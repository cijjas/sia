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
from core.algorithms.greedy_local import greedy_local
from core.algorithms.greedy_global import greedy_global
from core.models.state import State
from core.models.node import Node
import pandas as pd
import matplotlib.pyplot as plt

MAPS_PATH = "./maps"
OUTPUT_DIR = "output"

# Map user input to the corresponding functions
algorithm_map = {
    "BFS": bfs,
    "DFS": dfs,
    "A_STAR": a_star,
    "GREEDY_LOCAL": greedy_local,
    "GREEDY_GLOBAL": greedy_global,

}

# Map user input to the corresponding heuristic class
heuristics_map = {
    "Manhattan1": ManhattanDistance1(),
    "Manhattan2": ManhattanDistance2(),
    "Manhattan3": ManhattanDistance3(),
    "Euclidean": EuclideanDistance(),
    "DeadlockCorner": DeadlockCorner(),
    "Deadlock": Deadlock()
}

def create_data_set(maps, algorithm_configs, iterations_for_average, csv_file_name):
    total_executions = len(maps) * len(algorithm_configs) * iterations_for_average
    current_execution = 0

    with open(f'{OUTPUT_DIR}/{csv_file_name}', mode='w', newline='') as file:
        writer = csv.writer(file)
        csv_header = ['map', 'algorithm', 'heuristics_used', 'iteration', 'execution_time', 'expanded_nodes', 'frontier_nodes', 'success_or_failure', 'cost']

        writer.writerow(csv_header)

        for m in maps:
            map_data = parse_map(f'{MAPS_PATH}/{m}')
            initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'], map_data['spaces'])
            initial_state.init_deadlock_areas()
            initial_node = Node(initial_state, None, None, 0)

            for algo_config in algorithm_configs:
                algorithm_name = algo_config['name']
                heuristics = [heuristics_map[h] for h in algo_config['heuristics']]
                algorithm_function = algorithm_map.get(algorithm_name)

                if algorithm_function is None:
                    print(f"Invalid Algorithm: {algorithm_name}")
                    sys.exit(1)

                for i in range(iterations_for_average):
                    start_time = time.time()

                    if algorithm_name in ["A_STAR", "GREEDY_LOCAL", "GREEDY_GLOBAL"]:
                        solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node, heuristics)
                    else:
                        solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node)

                    execution_time = time.time() - start_time
                    row = [
                        m.split('.')[0],
                        algorithm_name,
                        "- ".join(str(h) if h is not None else "-" for h in heuristics),
                        i + 1, execution_time,
                        expanded_nodes,
                        frontier_nodes,
                        "SUCCESS" if solution_path is not None else "FAILURE",
                        len(solution_path) if solution_path else 0
                    ]
                    writer.writerow(row)

                    current_execution += 1
                    print(f"Progress: {current_execution}/{total_executions} executions completed ({(current_execution / total_executions) * 100:.2f}%)")

def create_comparison_data_set(map, algorithm_config, heuristics, iterations_for_average, csv_file_name):
    """ Create a data set for comparing heuristics under the same algorithm and map """
    total_executions = len(heuristics) * iterations_for_average
    current_execution = 0

    with open(f'{OUTPUT_DIR}/{csv_file_name}', mode='w', newline='') as file:
        writer = csv.writer(file)
        csv_header = ['map', 'algorithm', 'heuristics_used', 'iteration', 'execution_time', 'expanded_nodes', 'frontier_nodes', 'success_or_failure', 'cost']

        writer.writerow(csv_header)

        map_data = parse_map(f'{MAPS_PATH}/{map}')
        initial_state = State(map_data['walls'], map_data['goals'], map_data['boxes'], map_data['player'], map_data['spaces'])
        initial_state.init_deadlock_areas()
        initial_node = Node(initial_state, None, None, 0)

        algorithm_name = algorithm_config['name']
        algorithm_function = algorithm_map.get(algorithm_name)

        if algorithm_function is None:
            print(f"Invalid Algorithm: {algorithm_name}")
            sys.exit(1)

        for h in heuristics:
            for i in range(iterations_for_average):
                start_time = time.time()

                if algorithm_name in ["A_STAR", "GREEDY_LOCAL", "GREEDY_GLOBAL"]:
                    solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node, [h])
                else:
                    solution_path, expanded_nodes, frontier_nodes = algorithm_function(initial_node)

                execution_time = time.time() - start_time
                row = [
                    map.split('.')[0],
                    algorithm_name,
                    str(h),
                    i + 1, execution_time,
                    expanded_nodes,
                    frontier_nodes,
                    "SUCCESS" if solution_path is not None else "FAILURE",
                    len(solution_path) if solution_path else 0
                ]
                writer.writerow(row)

                current_execution += 1
                print(f"Progress: {current_execution}/{total_executions} executions completed ({(current_execution / total_executions) * 100:.2f}%)")


def main():
    if len(sys.argv) != 2:
        print("Choose a config file!")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

        executions = config['executions']
        for exec_config in executions:
            name = exec_config['output_file']
            data_gen = exec_config['data_generation']
            maps = data_gen['maps']
            algorithm_configs = data_gen['algorithms']
            iterations_for_average = data_gen['iterations_for_average']

            if 'compare' in exec_config['data_generation']:
                compare = exec_config['data_generation']['compare']
                if compare:
                    heuristics_name = algorithm_configs[0].get('heuristics', [])
                    heuristics = [heuristics_map[h] for h in heuristics_name]
                    create_comparison_data_set(maps[0], algorithm_configs[0], heuristics, iterations_for_average, name)
            else:
                create_data_set(maps, algorithm_configs, iterations_for_average, name)

    print("Data generation completed!")

if __name__ == "__main__":
    main()