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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"

def show_comparison_graphs(df, algorithms_to_compare):
    filtered_df = df[df['algorithm'].isin(algorithms_to_compare)]
    grouped_df = filtered_df.groupby(['algorithm']).agg({
        'expanded_nodes': 'mean',
        'execution_time': ['mean', 'std']
    }).reset_index()

    grouped_df.columns = ['algorithm', 'expanded_nodes_mean', 'execution_time_mean', 'execution_time_std']

    # Bar Chart for Expanded Nodes
    colors = plt.get_cmap('tab10')
    color_list = colors(np.linspace(0, 1, len(algorithms_to_compare)))
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['algorithm'], grouped_df['expanded_nodes_mean'], color=color_list)
    plt.title(f'Expanded Nodes for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Expanded Nodes')
    plt.show()

    # Bar Chart for Execution Time
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['algorithm'], grouped_df['execution_time_mean'],
            yerr=grouped_df['execution_time_std'], color=['red', 'orange'], capsize=5)
    plt.title(f'Execution Time for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (s)')
    plt.show()


def main():
    # all_algorithms.json
    # Differences in time and spatial complexity for all algorithms
    # Conclusion : BFS takes longer and expands more nodes than BFS but get to optimal, heuristics seem to be good...
    file_path = f'{OUTPUT_DIR}/all_algorithms.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "GREEDY_GLOBAL", "A_STAR", "DFS", "GREEDY_LOCAL"])

    # bfs_vs_dfs_rigged.json
    # Differences in time and spatial complexity for BFS and DFS in maps that ruin DFS preference
    # Conclusion : En ciertos casos expanden los mismos nodos e incluso DFS puede ser mas lento debido a la preference
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"])

    # bfs_vs_global_vs_a_star.json
    # Differences in time and spatial complexity for BFS, A_STAR and GLOBAL
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/bfs_vs_global_vs_a_star.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "GREEDY_GLOBAL", "A_STAR"])

    # bfs_vs_global_vs_a_star_bad_trade_off.json
    # Differences in time and spatial complexity for BFS, A_STAR and GLOBAL
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/bfs_vs_global_vs_a_star_bad_trade_off.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "GREEDY_GLOBAL", "A_STAR"])

    # dfs_vs_local.json
    # Differences in time and spatial complexity for DFS and LOCAL
    # Good trade off using Deadlock Corner Heuristic DFS
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/dfs_vs_local.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"])

    # dfs_vs_local_bad_trade_off.json
    # Differences in time and spatial complexity for DFS and LOCAL
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/dfs_vs_local_bad_trade_off.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"])

    # ------------------------------------------------------------

    # compare heuristics in different maps => some are better in some cases
    # inadmissible heuristic being really good => can be really good
    # compare all types of manhattans => which is better in which maps
    # combine heuristics, which combination is better
    # preprocesamiento FTW


if __name__ == "__main__":
    main()