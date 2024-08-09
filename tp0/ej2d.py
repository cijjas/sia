import json
import sys
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

LEVEL_MAX = 100
HEALTH_MAX = 1

HEALTH_ARRAY = [0.01, 0.25, 0.5, 0.75, 1]

# Teniendo en cuenta uno o dos pokemones distintos: ¿Que combinacion de condiciones
# (propiedades mutables) y pokebola conviene utilizar para capturarlos?

# We are currently only taking into consideration the HP of the pokemon. We also need to take into account the other parameters.

def load_2(pokemonName, pokeballs, directory, noise=0, reps=1000):

    # for the given pokemon, create a pokemon object array where the first pokemon has 1% of health and the second one has 100% of health

    pokemon_array = []
    print(HEALTH_ARRAY)
    print(len(HEALTH_ARRAY))

    for i in range(len(HEALTH_ARRAY)):
        pokemon = factory.create(pokemonName, LEVEL_MAX, StatusEffect.NONE, HEALTH_MAX*HEALTH_ARRAY[i])
        pokemon_array.append(pokemon)
    
    
    # Open the file once before the loop
    with open(f"{directory}/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokemon', 'Health%', 'Pokeball', 'Catch Success'])
        
        for j in range(len(HEALTH_ARRAY)):
            for pokeball in pokeballs:
                for rep in range(reps):
                    attempt_result = attempt_catch(pokemon_array[j], pokeball, noise)[0]
                    writer.writerow([pokemonName, f"{HEALTH_ARRAY[j]}", pokeball, attempt_result])



def analyze_2d():
    all_data = pd.DataFrame()

    for filename in os.listdir('output/2d'):
        data = pd.read_csv(f'output/2d/{filename}')
        pokemon_name = filename.split('.')[0]
        data['Pokemon'] = pokemon_name
        all_data = pd.concat([all_data, data], ignore_index=True)

    # hago la columna un booleano
    all_data['Catch Success'] = all_data['Catch Success'].astype(bool)

    # agrupo por pokeball y pokemon y saco la media de catch success
    grouped_data = all_data.groupby(['Pokeball', 'Pokemon', 'Health%'])['Catch Success'].mean().unstack()
    print(grouped_data)
    # Probabilidad promedio de captura por tipo de pokeball para cada pokemon
    ax = grouped_data.plot(kind='barh', figsize=(14, 8), title="2d. Probabilidad promedio de captura por tipo de pokeball para cada pokemon")
    ax.set_xlabel('Pokeball')
    ax.set_ylabel('Probabilidad de captura')
    ax.grid(True, linestyle='--')
    ax.legend(title='Pokemon')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Promedio de captura por tipo de pokeball para todos los pokemones con error estandar 
    mean_data = all_data.groupby('Pokeball')['Catch Success'].mean()
    std_err_data = all_data.groupby('Pokeball')['Catch Success'].std() 
    plt.figure(figsize=(10, 6))
    plt.bar(mean_data.index, mean_data, yerr=std_err_data, capsize=5, color='skyblue', label='Mean with SEM')
    plt.title('2d. Promedio de captura por tipo de pokeball para todos los pokemones con error estandar')
    plt.xlabel('Pokeball')
    plt.ylabel('Probabilidad de captura')
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

            load_2(config["pokemon"], config["pokeballs"], directory='output/2d', reps=100)
    
    analyze_2d()
