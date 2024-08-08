import json
import sys
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

LEVEL_MAX = 100
HEALTH_MAX = 1

def load(pokemonName, pokeballs, directory, noise=0, reps=1000):

    pokemon = factory.create( pokemonName,LEVEL_MAX, StatusEffect.NONE, HEALTH_MAX)

    with open(f"{directory}/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Catch Success'])
        for pokeball in pokeballs:
            for i in range(reps, 0, -1):  
                writer.writerow([pokeball, attempt_catch(pokemon, pokeball, noise)[0]])

def analyze_1a_bars():
    all_data = pd.DataFrame()

    # Load data from all CSV files in the directory
    for filename in os.listdir('output/1a'):
        data = pd.read_csv(f'output/1a/{filename}')
        pokemon_name = filename.split('.')[0]
        data['Pokemon'] = pokemon_name
        all_data = pd.concat([all_data, data], ignore_index=True)

    # hago la columna un booleano
    all_data['Catch Success'] = all_data['Catch Success'].astype(bool)

    # agrupo por pokeball y pokemon y saco la media de catch success
    grouped_data = all_data.groupby(['Pokeball', 'Pokemon'])['Catch Success'].mean().unstack()

    # Probabilidad promedio de captura por tipo de pokeball para cada pokemon
    ax = grouped_data.plot(kind='bar', figsize=(14, 8), title="Probabilidad promedio de captura por tipo de pokeball para cada pokemon")
    ax.set_xlabel('Pokeball')
    ax.set_ylabel('Probabilidad de captura')
    ax.grid(True, linestyle='--')
    ax.legend(title='Pokemon')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Promedio de captura por tipo de pokeball para todos los pokemones con error estandar 
    mean_data = all_data.groupby('Pokeball')['Catch Success'].mean()
    std_err_data = all_data.groupby('Pokeball')['Catch Success'].sem()  # Standard Error of the Mean
    plt.figure(figsize=(10, 6))
    plt.bar(mean_data.index, mean_data, yerr=std_err_data, capsize=5, color='skyblue', label='Mean with SEM')
    plt.title('Promedio de captura por tipo de pokeball para todos los pokemones con error estandar')
    plt.xlabel('Pokeball')
    plt.ylabel('Probabilidad de captura')
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


    # Varianza de las probabilidades de captura para cada pokeball
    variance_data = all_data.groupby('Pokeball')['Catch Success'].var()
    plt.figure(figsize=(10, 6))
    variance_data.plot(kind='line', color='tomato', title='Varianza de las probabilidades de captura para cada pokeball')
    plt.xlabel('Pokeball')
    plt.ylabel('Varianza')
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config_dir = sys.argv[1]
    pokemon_configs = os.listdir(config_dir)
    for pokemon_json in pokemon_configs:
        with open(f"{config_dir}/{pokemon_json}", "r") as f:
            config = json.load(f)

            # 1.a Para un pokemon en especifico calcular la media y desviacion estandar de la probabilidad de captura de 1000 intentos
            load(config["pokemon"], config["pokeballs"], "output/1a", 0.15)
    
    analyze_1a_bars()

