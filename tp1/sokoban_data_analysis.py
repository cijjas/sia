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

    # Plot for Expanded Nodes
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped_df['algorithm'], grouped_df['expanded_nodes_mean'], color=plt.cm.Paired.colors, capsize=5)
    plt.title(f'Expanded Nodes for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Expanded Nodes')
    
    # Adding text labels above bars for Expanded Nodes
    for bar in bars:
        yval = bar.get_height()
        plt.text((bar.get_x() + bar.get_width() / 2) , yval, f'{yval:.0f}', ha='center', va='bottom')

    plt.savefig(f'output/graphs/en_{output_file}.png')
    plt.close()

    # Plot for Execution Time
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped_df['algorithm'], grouped_df['execution_time_mean'], yerr=grouped_df['execution_time_std'], color=plt.cm.Accent.colors, capsize=5)
    plt.title(f'Execution Time for {", ".join(algorithms_to_compare)}')
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (s)')

    # Adding text labels above bars for Execution Time
    for bar in bars:
        yval = bar.get_height()
        plt.text((bar.get_x() + bar.get_width() / 2) +1/10, yval, f'{yval:.4f}', ha='center', va='bottom')

    plt.savefig(f'output/graphs/t_{output_file}.png')
    plt.close()

def compare_heuristics(data):
            # Group data by heuristics and calculate mean and standard deviation for execution time
        grouped_stats = data.groupby('heuristics_used').agg(
            mean_execution_time=pd.NamedAgg(column='execution_time', aggfunc='mean'),
            std_execution_time=pd.NamedAgg(column='execution_time', aggfunc='std'),
            mean_expanded_nodes=pd.NamedAgg(column='expanded_nodes', aggfunc='mean')
        ).reset_index()

        # Get the mean cost for each heuristic
        mean_cost = data.groupby('heuristics_used').agg(
            mean_cost=pd.NamedAgg(column='cost', aggfunc='mean')
        ).reset_index()

        # Plotting all three graphs
        fig, axs = plt.subplots(1, 3, figsize=(21, 5))

        # Graph 1: Cost by Heuristic
        axs[0].bar(mean_cost['heuristics_used'], mean_cost['mean_cost'], color='teal')
        axs[0].set_title('Costo por Heurística')
        axs[0].set_xlabel('Heurísticas')
        axs[0].set_ylabel('Costo Promedio')
        axs[0].set_xticks(range(len(mean_cost['heuristics_used'])))
        axs[0].set_xticklabels(mean_cost['heuristics_used'], rotation=90)

        # Graph 2: Mean Execution Time with Standard Deviation
        axs[1].bar(grouped_stats['heuristics_used'], grouped_stats['mean_execution_time'], 
                yerr=grouped_stats['std_execution_time'], color='coral', alpha=0.7)
        axs[1].set_title('Tiempo Promedio de Ejecución con STD')
        axs[1].set_xlabel('Heurísticas')
        axs[1].set_ylabel('Tiempo de Ejecución (s)')
        axs[1].set_xticks(range(len(mean_cost['heuristics_used'])))
        axs[1].set_xticklabels(mean_cost['heuristics_used'], rotation=90)

        # Graph 3: Expanded Nodes by Heuristic
        axs[2].bar(grouped_stats['heuristics_used'], grouped_stats['mean_expanded_nodes'], color='skyblue')
        axs[2].set_title('Nodos Expandidos por Heurística')
        axs[2].set_xlabel('Heurísticas')
        axs[2].set_ylabel('Nodos Expandidos')
        axs[2].set_xticks(range(len(mean_cost['heuristics_used'])))
        axs[2].set_xticklabels(mean_cost['heuristics_used'], rotation=90)

        plt.tight_layout()
        plt.show()

def main():
    # all_algorithms.json
    # Conclusion : BFS takes longer and expands more nodes than BFS but get to optimal, heuristics seem to be good...
    file_path = f'{OUTPUT_DIR}/all_algorithms.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "GREEDY_GLOBAL", "A_STAR", "DFS", "GREEDY_LOCAL"], "all_algorithms")

    # bfs_vs_dfs.json
    # Conclusion : Usual case where BFS expands more nodes and takes more time than DFS
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"], "bfs_vs_dfs")

    # bfs_vs_dfs_rigged.json
    # Conclusion : En ciertos casos expanden los mismos nodos e incluso DFS puede ser mas lento debido a la preference
    file_path = f'{OUTPUT_DIR}/bfs_vs_dfs_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "DFS"], "bfs_vs_dfs_rigged")

    # bfs_vs_a_star.json
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/bfs_vs_a_star.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"], "bfs_vs_a_star")

    # bfs_vs_a_star_rigged.json
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/bfs_vs_a_star_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["BFS", "A_STAR"], "bfs_vs_a_star_rigged")

    # dfs_vs_local.json
    # Conclusion : heuristics seems to be the way to go
    file_path = f'{OUTPUT_DIR}/dfs_vs_local.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"], "dfs_vs_local")

    # dfs_vs_local_rigged.json
    # Conclusion : heuristic's added time complexity is not worth the spatial savings
    file_path = f'{OUTPUT_DIR}/dfs_vs_local_rigged.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        show_comparison_graphs(df, ["DFS", "GREEDY_LOCAL"], "dfs_vs_local_rigged")

    #### These do not expose the right information, they are grouping by algorithm but should be grouping by heuristic!

    # admissible_vs_inadmissible.json
    # Conclusion : inadmissible can be really good in specific situations
    file_path = f'{OUTPUT_DIR}/admissible_vs_inadmissible.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        compare_heuristics(data)
        

    # heuristic_permutation.json
    # Conclusion : heuristics performance in a scenario
    file_path = f'{OUTPUT_DIR}/heuristic_permutation.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        compare_heuristics(data)

    # heuristic_permutation_2.json
    # Conclusion : heuristics performance varies according to the scenario
    file_path = f'{OUTPUT_DIR}/heuristic_permutation_2.csv'
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        compare_heuristics(data)


    #file_path = f'{OUTPUT_DIR}/deadlock_cmp_a_star.csv'
    #if os.path.exists(file_path):
    #    df = pd.read_csv(file_path)
    #    show_heuristics_comparison_graphs(df, "A_STAR", ["Deadlock", "DeadlockCorner"])

    # ------------------------------------------------------------

    file_path = f'{OUTPUT_DIR}/smarthattan2.csv'
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
            "Smarthattan": "Smarthattan",
            "M1": "M1",
            "M2": "M2",
            "M3": "M3",
        }
        # Bar plot for Expanded Nodes
        bars1 = ax1.bar(grouped['heuristics_used'],expanded_nodes['expanded_nodes'],
                capsize=5, color='blue', alpha=0.7)
        ax1.set_title('Nodos Expandidos vs Heurísticas')
        ax1.set_xlabel('Heurísticas')
        ax1.set_ylabel('Nodos Expandidos')
        ax1.set_xticks(range(len(grouped['heuristics_used'])))
        ax1.set_xticklabels([new_names[name] for name in grouped['heuristics_used']])

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