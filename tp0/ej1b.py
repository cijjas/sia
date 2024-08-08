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


def analyze_1b_bars():
    # we are going to make a bar plot
    all_data = pd.DataFrame()

    # Load data from all CSV files in the directory
    for filename in os.listdir('output/1b'):
        data = pd.read_csv(f'output/1b/{filename}')
        pokemon_name = filename.split('.')[0]
        data['Pokemon'] = pokemon_name
        all_data = pd.concat([all_data, data], ignore_index=True)
    
    # hago la columna un booleano
    all_data['Catch Success'] = all_data['Catch Success'].astype(bool)

    # agrupo por pokeball y pokemon y saco la media de catch success
    grouped_data = all_data.groupby(['Pokeball', 'Pokemon'])['Catch Success'].mean().unstack()

    catch_success_for_pokeball = grouped_data.loc['pokeball']
    grouped_data = grouped_data.div(catch_success_for_pokeball, axis=1)
    print(catch_success_for_pokeball)

    # Probabilidad promedio de captura por tipo de pokeball para cada pokemon
    ax = grouped_data.plot(kind='bar', figsize=(14, 8), title="Probabilidad promedio de captura por tipo de pokeball para cada pokemon")
    ax.set_xlabel('Pokeball')
    ax.set_ylabel('Probabilidad de captura')
    ax.grid(True, linestyle='--')
    ax.legend(title='Pokemon')
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
            load(config["pokemon"], config["pokeballs"], "output/1b", 0, 100000)
    
    analyze_1b_bars()

