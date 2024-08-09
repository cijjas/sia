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



if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config_dir = sys.argv[1]
    pokemon_configs = os.listdir(config_dir)
    for pokemon_json in pokemon_configs:
        with open(f"{config_dir}/{pokemon_json}", "r") as f:
            config = json.load(f)
            load_2b(config["pokemon"], config["pokeballs"])
    analyze_2b()

