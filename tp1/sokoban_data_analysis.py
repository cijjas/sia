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

def show_heuristics_comparison_graphs(df, algorithm, heuristics_to_compare, output_file="pone_nombre"):
    os.makedirs('output/graphs', exist_ok=True)

    filtered_df = df[(df['algorithm'] == algorithm) & (df['heuristics_used'].isin(heuristics_to_compare))]
    grouped_df = filtered_df.groupby(['heuristics_used']).agg({
        'expanded_nodes': 'mean',
        'execution_time': ['mean', 'std']
    }).reset_index()

    grouped_df.columns = ['heuristics_used', 'expanded_nodes_mean', 'execution_time_mean', 'execution_time_std']

    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['heuristics_used'], grouped_df['expanded_nodes_mean'], color=plt.cm.Paired.colors, capsize=5)
    plt.title(f'Nodos expandidos para las heurísticas {", ".join(heuristics_to_compare)} con el algoritmo {algorithm}')
    plt.xlabel('Heuristica')
    plt.ylabel('Nodos expandidos')
    plt.savefig(f'output/graphs/en_{output_file}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(grouped_df['heuristics_used'], grouped_df['execution_time_mean'],
            yerr=grouped_df['execution_time_std'], color=plt.cm.Accent.colors, capsize=5)
    plt.title(f'Tiempo de ejecución del algoritmo {algorithm} con las heurísticas {", ".join(heuristics_to_compare)}')
    plt.xlabel('Heuristica')
    plt.ylabel('Tiempo de ejecución [s]')
    plt.savefig(f'output/graphs/t_{output_file}.png')
    plt.close()

def main():
    # all_algorithms.json
    # Differences in time and spatial complexity for all algorithms
    # Conclusion : BFS takes longer and expands more nodes than BFS but get to optimal, heuristics seem to be good...
    file_path = f'{OUTPUT_DIR}/all_algorithms.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "GREEDY_GLOBAL", "A_STAR", "DFS", "GREEDY_LOCAL"])

    # bfs_vs_dfs.json
    # Differences in time and spatial complexity for BFS and DFS in maps that ruin DFS preference
    # Conclusion : Usual case where BFS expands more nodes and takes more time than DFS
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"], "bfs_vs_dfs")

    # bfs_vs_dfs_rigged.json
    # Differences in time and spatial complexity for BFS and DFS in maps that ruin DFS preference
    # Conclusion : En ciertos casos expanden los mismos nodos e incluso DFS puede ser mas lento debido a la preference
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"], "bfs_vs_dfs")

    # bfs_vs_a_star.json
    # Differences in time and spatial complexity for BFS, A_STAR and GLOBAL
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/bfs_vs_a_star.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"])

    # bfs_vs_a_star_rigged.json
    # Differences in time and spatial complexity for BFS, A_STAR and GLOBAL
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/bfs_vs_a_star_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"])

    # dfs_vs_local.json
    # Differences in time and spatial complexity for DFS and LOCAL
    # Good trade off using Deadlock Corner Heuristic DFS
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/dfs_vs_local.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"])

    # dfs_vs_local_rigged.json
    # Differences in time and spatial complexity for DFS and LOCAL
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/dfs_vs_local_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"])

    # deadlock_cmp_a_star.json
    # Differences in time and spatial complexity for A_STAR with corner deadlocks vs corner deadlocks + wall deadlocks
    # Conclusion : The maps which have wall deadlock areas perform better with the added heuristic
    
    #file_path = f'{OUTPUT_DIR}/deadlock_cmp_a_star.csv'
    #if os.path.exists(file_path):
    #    df = pd.read_csv(file_path)
    #    show_heuristics_comparison_graphs(df, "A_STAR", ["Deadlock", "DeadlockCorner"])

    # ------------------------------------------------------------

    # inadmissible heuristic being really good => can be really good
    # preprocesamiento FTW
    # Tratar las heuristicas de formas ortogonales, hacer preanalisis del mapa, por ejemplo en que cuadrante se encuentra la mayoria de lostarges para definir la preferencia del algoritmo


    file_path = f'{OUTPUT_DIR}/smarthattan.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Group data by heuristics and calculate mean and standard deviation
        grouped = df.groupby('heuristics_used').agg(
            mean_execution_time=pd.NamedAgg(column='execution_time', aggfunc='mean'),
            std_execution_time=pd.NamedAgg(column='execution_time', aggfunc='std'),
        ).reset_index()

        # Create a figure with subplots for bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        expanded_nodes = df.groupby('heuristics_used')['expanded_nodes'].first().reset_index()


        new_names = {
            "ManhattanDistance1": "M1",
            "ManhattanDistance2": "M2",
            "ManhattanDistance3": "M3",
            "Smarthattan": "Smarthattan",
        }
        # Bar plot for Expanded Nodes
        bars1 = ax1.bar(grouped['heuristics_used'],expanded_nodes['expanded_nodes'],
                capsize=5, color='blue', alpha=0.7)
        ax1.set_title('Nodos Expandidos vs Heurísticas')
        ax1.set_xlabel('Heurísticas')
        ax1.set_ylabel('Nodos Expandidos')
        ax1.set_xticks(range(len(grouped['heuristics_used'])))
        ax1.set_xticklabels([new_names[name] for name in grouped['heuristics_used']])
        ax1.set_ylim(32000, max(expanded_nodes['expanded_nodes']) + 1000)  # Set y-axis to start at 30000

        # Adding text labels on bars
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

        # Bar plot for Execution Time
        bars2 = ax2.bar(grouped['heuristics_used'], grouped['mean_execution_time'],
                yerr=grouped['std_execution_time'], capsize=5, color='red', alpha=0.7)
        ax2.set_title('Tiempo Promedio de Ejecución vs Heurísticas')
        ax2.set_xlabel('Heurísticas')
        ax2.set_ylabel('Tiempo de Ejecución (s)')
        ax2.set_xticks(range(len(grouped['heuristics_used'])))
        ax2.set_xticklabels([new_names[name] for name in grouped['heuristics_used']])

        # Adding text labels on bars
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), va='bottom')  # Format for more precision

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()