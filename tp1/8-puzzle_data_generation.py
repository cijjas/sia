import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output"
GRAPH_DIR = "graphs"

def generate_graphs(algorithm_name):
    file_path = f'{OUTPUT_DIR}/{algorithm_name}_8puzzle_results.csv'
    if not os.path.exists(file_path):
        print(f"No data found for {algorithm_name}")
        return

    df = pd.read_csv(file_path)

    # Average Time Graph
    avg_time_df = df.groupby('heuristic').agg(
        mean_execution_time=pd.NamedAgg(column='execution_time', aggfunc='mean'),
        std_execution_time=pd.NamedAgg(column='execution_time', aggfunc='std')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.barh(avg_time_df['heuristic'], avg_time_df['mean_execution_time'], xerr=avg_time_df['std_execution_time'], capsize=5)
    plt.title(f'Average Execution Time for {algorithm_name}')
    plt.xlabel('Execution Time (s)')
    plt.ylabel('Heuristic')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.4f}', va='center')
    plt.savefig(f'{OUTPUT_DIR}/{GRAPH_DIR}/{algorithm_name}_avg_time.png')
    plt.close()

    # Expanded Nodes Graph
    expanded_nodes_df = df.groupby('heuristic').agg(
        mean_expanded_nodes=pd.NamedAgg(column='expanded_nodes', aggfunc='mean')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.barh(expanded_nodes_df['heuristic'], expanded_nodes_df['mean_expanded_nodes'], capsize=5)
    plt.title(f'Expanded Nodes for {algorithm_name}')
    plt.xlabel('Expanded Nodes')
    plt.ylabel('Heuristic')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.0f}', va='center')
    plt.savefig(f'{OUTPUT_DIR}/{GRAPH_DIR}/{algorithm_name}_expanded_nodes.png')
    plt.close()

    # Total Movements Graph
    total_movements_df = df.groupby('heuristic').agg(
        mean_total_movements=pd.NamedAgg(column='total_movements', aggfunc='mean')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.barh(total_movements_df['heuristic'], total_movements_df['mean_total_movements'], capsize=5)
    plt.title(f'Total Movements for {algorithm_name}')
    plt.xlabel('Total Movements')
    plt.ylabel('Heuristic')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.0f}', va='center')
    plt.savefig(f'{OUTPUT_DIR}/{GRAPH_DIR}/{algorithm_name}_total_movements.png')
    plt.close()

def generate_output_path():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

def main():
    generate_output_path()
    algorithms = ["a_star", "global_greedy"]
    for algorithm in algorithms:
        generate_graphs(algorithm)

if __name__ == "__main__":
    main()