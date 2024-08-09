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

def load(pokemonName, pokeballs, directory, noise=0, reps=1000):

    pokemon = factory.create( pokemonName,LEVEL_MAX, StatusEffect.NONE, HEALTH_MAX)

    with open(f"{directory}/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Catch Success'])
        for pokeball in pokeballs:
            for i in range(reps, 0, -1):  
                writer.writerow([pokeball, attempt_catch(pokemon, pokeball, noise)[0]])

def analyze_1a():
    all_data = pd.DataFrame()

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
    ax = grouped_data.plot(kind='bar', figsize=(14, 8), title="1a. Probabilidad promedio de captura por tipo de pokeball para cada pokemon")
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
    plt.title('1a. Promedio de captura por tipo de pokeball para todos los pokemones con error estandar')
    plt.xlabel('Pokeball')
    plt.ylabel('Probabilidad de captura')
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


    # Varianza de las probabilidades de captura para cada pokeball
    variance_data = all_data.groupby('Pokeball')['Catch Success'].var()
    plt.figure(figsize=(10, 6))
    variance_data.plot(kind='line', color='tomato', title='1a. Varianza de las probabilidades de captura para cada pokeball', marker='o')
    plt.xlabel('Pokeball')
    plt.ylabel('Varianza')
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_1b():
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

def analyze_2a():
    # Load data from CSV
    data = pd.read_csv('output/2a.csv')
    pokemon = data['Pokemon'].unique()
    title = 'Efecto de la vida (HP) en la probabilidad de captura para ' + ', '.join(pokemon)
    data['Probability'] = pd.to_numeric(data['Probability'], errors='coerce')
    
    data.set_index(['Current HP', 'Pokeball'], inplace=True)
    hp_effect = data['Probability'].unstack()

    # Plotting
    plt.figure(figsize=(12, 8))
    markers = ['o', '^', 's', 'p'] 
    line_styles = ['-', '--', '-.', ':']
    for idx, pokeball in enumerate(hp_effect.columns):
        plt.plot(hp_effect.index, hp_effect[pokeball], 
                 marker=markers[idx % len(markers)], 
                 linestyle=line_styles[idx % len(line_styles)], 
                 label=str(pokeball))
    plt.title(title )
    plt.xlabel('HP del pokemon')
    plt.ylabel('Probabilidad de captura')
    plt.legend(title='Pokeball', loc='upper right')  
    plt.grid(True)
    plt.show()


def load_2a(pokemonName, pokeballs, noise=0):
    with open('output/2a.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Current HP'])
        for pokeball in pokeballs:
            for i in range(100, 0, -1):  
                pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/100)
                writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], i/100])

def load_2b(pokemonName, pokeballs, noise=0):
    with open(f"output/2b/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Current HP'])
        for pokeball in pokeballs:
            for i in range(100, 0, -1):  
                pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/100)
                writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], i/100])


def analyze_2b():
    # Load data from CSV
    data = pd.read_csv(f"output/2b/{os.listdir('output/2b').pop()}")
    pokemon = data['Pokemon'].unique()
    title = '2b. Efecto de la vida (HP) en la probabilidad de captura para ' + ', '.join(pokemon)
    data['Probability'] = pd.to_numeric(data['Probability'], errors='coerce')
    
    data.set_index(['Current HP', 'Pokeball'], inplace=True)
    hp_effect = data['Probability'].unstack()

    # Plotting
    plt.figure(figsize=(12, 8))
    markers = ['o', '^', 's', 'p'] 
    line_styles = ['-', '--', '-.', ':']
    for idx, pokeball in enumerate(hp_effect.columns):
        plt.plot(hp_effect.index, hp_effect[pokeball], 
                 marker=markers[idx % len(markers)], 
                 linestyle=line_styles[idx % len(line_styles)], 
                 label=str(pokeball))
    plt.title(title )
    plt.xlabel('Porcentaje de HP de Pokémon')
    plt.ylabel('Probabilidad de captura')
    plt.legend(title='Pokeball', loc='upper right')  
    plt.grid(True)
    plt.show()


def load_2a(pokemonName, pokeballs, noise=0):
    with open(f"output/2a/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Status'])
        for pokeball in pokeballs:
            for status in StatusEffect.__members__.values():  
                pokemon = factory.create( pokemonName, LEVEL_MAX, status, HEALTH_MAX)
                writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], status._name_])

def analyze_2a():
    data = pd.read_csv(f"output/2a/{os.listdir('output/2a').pop()}")
    pokemon = data['Pokemon'].unique()
    title = '2a. Efecto de la salud sobre la efectividad de captura ' + ', '.join(pokemon)
    data['Probability'] = pd.to_numeric(data['Probability'], errors='coerce')

    data.set_index(['Status', 'Pokeball'], inplace=True) 
    hp_effect = data['Probability'].unstack()

    plt.figure(figsize=(12, 8))
    markers = ['o', '^', 's', 'p'] 
    line_styles = ['-', '--', '-.', ':']
    for idx, pokeball in enumerate(hp_effect.columns):
        plt.plot(hp_effect.index, hp_effect[pokeball], 
                 marker=markers[idx % len(markers)], 
                 linestyle=line_styles[idx % len(line_styles)], 
                 label=str(pokeball))
    plt.title(title )
    plt.xlabel('Estado de salud del Pokemon')
    plt.ylabel('Probabilidad de captura')
    plt.legend(title='Pokeball', loc='upper center')  
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config_dir = sys.argv[1]
    pokemon_configs = os.listdir(config_dir)
    for pokemon_json in pokemon_configs:
        with open(f"{config_dir}/{pokemon_json}", "r") as f:
            config = json.load(f)

            load(config["pokemon"], config["pokeballs"], directory='output/1a')
            load(config["pokemon"], config["pokeballs"], directory='output/1b', reps=10000)

            load_2a(config["pokemon"], config["pokeballs"])
            load_2b(config["pokemon"], config["pokeballs"])
    
    analyze_1a()
    analyze_1b()
    analyze_2a()
    analyze_2b()
