import os
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl # type: ignore
from openpyxl.styles import PatternFill # type: ignore
from openpyxl.utils.dataframe import dataframe_to_rows # type: ignore

OUTPUT_DIR = "../output"
SELECTION_METHOD_COMPARISON_FILE_NAME = "selection_method_comparison.csv"
DATA_DIR = "../output/data"

def selection_method_comparison():

    print("Generando heatmap para comparación de métodos de selección de padres y reemplazo en base al fitness promedio de la generación 100")
    
    # Cargar el CSV
    df = pd.read_csv(os.path.join(OUTPUT_DIR, SELECTION_METHOD_COMPARISON_FILE_NAME))

    # Crear una tabla pivot para tener métodos de padres en columnas, métodos de reemplazo en filas
    # strength,dexterity,intelligence,vigor,constitution,height,fitness,average_fitness,generation,parent_selection_method,replacement_selection_method
    pivot_table = df.pivot_table(index='replacement_selection_method', columns='parent_selection_method', values='average_fitness', aggfunc='mean')
    
    # Exportar a Excel
    excel_filename = os.path.join(DATA_DIR, 'heatmap.xlsx')

    # Crear directorio si no existe
    os.makedirs(DATA_DIR, exist_ok=True)
    pivot_table.to_excel(excel_filename)

    # Cargar el archivo de Excel para formatear el heatmap
    wb = openpyxl.load_workbook(excel_filename)
    ws = wb.active

    # Crear el heatmap con formato condicional
    # Obtener los rangos de los valores de fitness
    min_val = pivot_table.min().min()
    max_val = pivot_table.max().max()

    # Aplicar colores en función del valor de fitness
    for row in ws.iter_rows(min_row=2, min_col=2, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            if isinstance(cell.value, (float, int)):  # Solo aplicar formato a celdas con números
                # Normalizar el valor
                normalized_value = (cell.value - min_val) / (max_val - min_val)
                # Generar el color (del rojo al verde)
                red = int(255 * (1 - normalized_value))
                green = int(255 * normalized_value)
                fill_color = PatternFill(start_color=f'{red:02X}{green:02X}00', end_color=f'{red:02X}{green:02X}00', fill_type='solid')
                cell.fill = fill_color

    # Guardar el archivo de Excel con el heatmap
    wb.save(excel_filename)
    print(f"El heatmap se guardó en {excel_filename}")

def main():
    selection_method_comparison()

if __name__ == "__main__":
    main()
