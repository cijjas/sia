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

OUTPUT_DIR = "output"

def show_comparison_graphs(df, algorithms_to_compare, output_file="pone_nombre"):
    os.makedirs('output/graphs', exist_ok=True)
    
    filtered_df = df[df['algorithm'].isin(algorithms_to_compare)]
    grouped_df = filtered_df.groupby(['algorithm']).agg({
        'expanded_nodes': 'mean',
        'execution_time': ['mean', 'std']
    }).reset_index()

    grouped_df.columns = ['algorithm', 'expanded_nodes_mean', 'execution_time_mean', 'execution_time_std']

    # https://matplotlib.org/stable/users/explain/colors/colormaps.html

    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['algorithm'], grouped_df['expanded_nodes_mean'], color=plt.cm.Paired.colors, capsize=5)
    plt.title(f'Expanded Nodes for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Expanded Nodes')
    plt.savefig(f'output/graphs/en_{output_file}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['algorithm'], grouped_df['execution_time_mean'],
            yerr=grouped_df['execution_time_std'], color=plt.cm.Accent.colors, capsize=5)
    plt.title(f'Execution Time for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (s)')
    plt.savefig(f'output/graphs/t_{output_file}.png')
    plt.close()

def main():
    # Generate analytics

    # Issues

    # 3. Cost :
    #   3.1 BFS get optimal solution, A_STAR with admissible heuristic also , DFS does not,, etc
    #   3.2 Non Admisible Heuristics that can be really good for certain maps
    #       Preprocesamiento del mapa para elegir correctamente las heuristicas que valen la pena (Ambiente finito nose que)
    # Comparacion de manhattans

    # Frontier Nodes : ???
    # success_failure : ???

    # bfs_vs_dfs.json
    # 1. Differences in time and spatial complexity for BFS and DFS
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"], "bfs_vs_dfs")

    # local_vs_global.json
    # 2. Differences in time and spatial complexity for Greedy Local and Greedy Global
    file_path = f'{OUTPUT_DIR}/greedy_local_vs_greedy_global.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["GREEDY_LOCAL", "GREEDY_GLOBAL"])

    # heuristic_good.json
    # 3. Good trade off using Deadlock Corner Heuristic
    file_path = f'{OUTPUT_DIR}/heuristic_good.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"])

    # heuristic_bad.json
    # 4. Bad trade off using Deadlock Corner Heuristic
    file_path = f'{OUTPUT_DIR}/heuristic_bad.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"])

    # heuristic_comparison.json
    # 5. Compare the efficiency of the heuristics in different maps
    file_path = f'{OUTPUT_DIR}/heuristic_comparison.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"])

    # 6. Comparison right preference vs left preference vs center preference vs random preference (can we change an algorithm to do that?)

if __name__ == "__main__":
    main()