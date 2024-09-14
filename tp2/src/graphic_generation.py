import os
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd
import openpyxl # type: ignore
from openpyxl.styles import PatternFill # type: ignore
from openpyxl.utils.dataframe import dataframe_to_rows # type: ignore
from pandas.core.frame import DataFrame
import seaborn as sns

OUTPUT_DIR = "../output"
SELECTION_METHOD_COMPARISON_AVG_FILE_NAME = "selection_method_comparison.csv"
SELECTION_METHOD_COMPARISON_BEST_FILE_NAME = "selection_method_comparison.csv"
SELECTION_METHOD_COMPARISON_AVG_FILE_NAME_2 = "selection_method_comparison_2.csv"
SELECTION_METHOD_COMPARISON_BEST_FILE_NAME_2 = "selection_method_comparison_2.csv"
SELECTION_RATE_FILE_NAME = "selection_rate_comparison.csv"
DATA_DIR = "../output/data"

def selection_method_comparison(input_dir, is_avg=True):
    print("Generando heatmap para comparación de métodos de selección de padres y reemplazo en base al fitness promedio de la generación 100")
    
    # Cargar el CSV
    df = pd.read_csv(os.path.join(input_dir, SELECTION_METHOD_COMPARISON_AVG_FILE_NAME if is_avg else SELECTION_METHOD_COMPARISON_BEST_FILE_NAME))

    # Truncar los nombres de los métodos de selección a las dos primeras letras
    df['parent_selection_method'] = df['parent_selection_method'].apply(lambda x: x[:2])
    df['replacement_selection_method'] = df['replacement_selection_method'].apply(lambda x: x[:2])

    # Crear un pivot table basado en el valor de is_avg
    if is_avg:
        pivot_table = df.pivot_table(index='replacement_selection_method', columns='parent_selection_method', values='fitness', aggfunc='mean')
    else:
        pivot_table = df.pivot_table(index='replacement_selection_method', columns='parent_selection_method', values='fitness', aggfunc='max')

    # Verificar si el DataFrame está vacío
    if pivot_table.empty:
        print("Advertencia: El DataFrame resultante está vacío. No se generará el heatmap.")
        return

    # Generar el heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot_table, annot=False, cmap='viridis', cbar_kws={'label': 'Fitness'})
    plt.title('Heatmap de Métodos de Selección' + (' (Promedio)' if is_avg else ' (Máximo)'))
    plt.xlabel('Método de Selección de Padres')
    plt.ylabel('Método de Selección de Reemplazo')
    plt.tight_layout()

    # Guardar el heatmap como imagen
    plt.savefig(os.path.join(DATA_DIR, 'heatmap_avg.png' if is_avg else 'heatmap_best.png'))
    plt.close()
    print(f"El heatmap se guardó en {os.path.join(DATA_DIR, 'heatmap_avg.png' if is_avg else 'heatmap_best.png')}")

def selection_method_comparison_2(input_dir, is_avg=True):
    print("Generando heatmap para comparación de métodos de selección de padres y reemplazo en base al fitness promedio de la generación 100")

    # Cargar el CSV
    df = pd.read_csv(os.path.join(input_dir, SELECTION_METHOD_COMPARISON_AVG_FILE_NAME_2 if is_avg else SELECTION_METHOD_COMPARISON_BEST_FILE_NAME_2))

    # Truncar los nombres de los métodos de selección a las dos primeras letras
    df['parent_selection_method_1'] = df['parent_selection_method_1'].apply(lambda x: x[:2])
    df['parent_selection_method_2'] = df['parent_selection_method_2'].apply(lambda x: x[:2])
    df['replacement_selection_method_1'] = df['replacement_selection_method_1'].apply(lambda x: x[:2])
    df['replacement_selection_method_2'] = df['replacement_selection_method_2'].apply(lambda x: x[:2])

    # Crear un pivot table basado en el valor de is_avg
    if is_avg:
        pivot_table = df.pivot_table(index=['replacement_selection_method_1', 'replacement_selection_method_2'], columns=['parent_selection_method_1', 'parent_selection_method_2'], values='fitness', aggfunc='mean')
    else:
        pivot_table = df.pivot_table(index=['replacement_selection_method_1', 'replacement_selection_method_2'], columns=['parent_selection_method_1', 'parent_selection_method_2'], values='fitness', aggfunc='max')

    # Verificar si el DataFrame está vacío
    if pivot_table.empty:
        print("Advertencia: El DataFrame resultante está vacío. No se generará el heatmap.")
        return

    # Generar el heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=False, cmap='viridis', cbar_kws={'label': 'Fitness'})
    plt.title('Heatmap de Métodos de Selección' + (' (Promedio)' if is_avg else ' (Máximo)'), fontsize=20)
    plt.xlabel('Método de Selección de Padres', fontsize=16)
    plt.ylabel('Método de Selección de Reemplazo', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Fitness', fontsize=16)
    plt.tight_layout()

    # Guardar el heatmap como imagen
    plt.savefig(os.path.join(DATA_DIR, 'heatmap_avg_2.png' if is_avg else 'heatmap_best_2.png'))
    plt.close()
    print(f"El heatmap se guardó en {os.path.join(DATA_DIR, 'heatmap_avg_2.png' if is_avg else 'heatmap_best_2.png')}")

#example
#strength,dexterity,intelligence,vigor,constitution,height,fitness,generation,selection_rate
#43,44,44,32,35,1.94,33.28143277855617,10000,1
def selection_rate_comparison(input_dir, is_avg=True):
    print("Generando gráfico de líneas para comparación de tasas de selección en base al fitness promedio de la generación 10000")
    
    # Cargar el CSV
    df = pd.read_csv(os.path.join(input_dir, SELECTION_RATE_FILE_NAME))

    # Crear un pivot table
    pivot_table = df.pivot_table(index='selection_rate', values='fitness', aggfunc='mean') if is_avg else df.pivot_table(index='selection_rate', values='fitness', aggfunc='max')

    # Interpolación suave
    x = pivot_table.index.values
    y = pivot_table['fitness'].values
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    # Generar el heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=False, cmap='viridis', cbar_kws={'label': 'Fitness'})
    plt.title('Heatmap de Métodos de Selección 2' + (' (Promedio)' if is_avg else ' (Máximo)'), fontsize=20)
    plt.xlabel('Método de Selección de Padres 1 y 2', fontsize=16)
    plt.ylabel('Método de Selección de Reemplazo 1 y 2', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Fitness', fontsize=16)
    plt.tight_layout()

    # Guardar el gráfico
    plt.savefig(os.path.join(DATA_DIR, 'selection_rate_comparison_avg.png' if is_avg else 'selection_rate_comparison_max.png'))
    plt.close()
    print(f"El gráfico de líneas se guardó en {os.path.join(DATA_DIR, 'selection_rate_comparison_avg.png' if is_avg else 'selection_rate_comparison_max.png')}")

def mutation_rate_comparison(input_dir, is_avg=True):
    print("Generando gráfico de barras para comparación de tasas de mutación en base al fitness promedio de la generación 100")
    
    # Cargar el CSV
    df = pd.read_csv(os.path.join(input_dir, "mutation_rate_comparison.csv"))

    # Crear un pivot table
    pivot_table = df.pivot_table(index='mutation_rate', values='fitness', aggfunc='mean') if is_avg else df.pivot_table(index='mutation_rate', values='fitness', aggfunc='max')

    # Generar el gráfico de barras
    pivot_table.plot(kind='bar', legend=False)
    plt.xlabel('Mutation Rate')
    plt.ylabel('Fitness')
    plt.title('Mutation Rate vs Fitness' + (' (Average)' if is_avg else ' (Max)'))
    plt.tight_layout()

    # Guardar el gráfico
    plt.savefig(os.path.join(DATA_DIR, 'mutation_rate_comparison_avg.png' if is_avg else 'mutation_rate_comparison_max.png'))
    plt.close()
    print(f"El gráfico de barras se guardó en {os.path.join(DATA_DIR, 'mutation_rate_comparison_avg.png' if is_avg else 'mutation_rate_comparison_max.png')}")

def main():
    selection_method_comparison(OUTPUT_DIR, is_avg=True)
    selection_method_comparison(OUTPUT_DIR, is_avg=False)
    selection_method_comparison_2(OUTPUT_DIR, is_avg=True)
    selection_method_comparison_2(OUTPUT_DIR, is_avg=False)
    mutation_rate_comparison(OUTPUT_DIR, is_avg=True)
    selection_rate_comparison(OUTPUT_DIR, is_avg=True)

if __name__ == "__main__":
    main()
