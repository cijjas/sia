import json
import sys
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect

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
                pokemon = factory.create( pokemonName, 100, StatusEffect.NONE, i/100)
                writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], i/100])


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config_dir = sys.argv[1]
    pokemon_configs = os.listdir(config_dir)
    for pokemon_json in pokemon_configs:
        with open(f"{config_dir}/{pokemon_json}", "r") as f:
            config = json.load(f)
            # 2.a Para un pokemon en especifico calcular la probabilidad de captura para cada pokeball en funcion de la vida del pokemon
            load_2a(config["pokemon"], config["pokeballs"], 0.15)
    analyze_2a()
    