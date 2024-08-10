import json
import sys
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect
from scipy.interpolate import make_interp_spline

LEVEL_MAX = 100
HEALTH_MAX = 1
SAVE_PATH = 'output/analyze/'
EJ1A = '1a'
EJ1B = '1b'
EJ2A = '2a'
EJ2B = '2b'
EJ2C = '2c'

def load_2a_health(pokemonName, pokeballs, noise=0):
    with open('output/2a/health/2a.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Captured', 'Current HP'])
        divisions = 20
        tries = 1000
        for pokeball in pokeballs:
            for i in range(divisions, 0, -1):
                for j in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/divisions)
                    writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], attempt_catch(pokemon, pokeball, noise)[0], i/divisions])

def load_2a_status(pokemonName, pokeballs, noise=0):
    with open(f"output/2a/status/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Captured', 'Status'])
        tries = 1000
        for pokeball in pokeballs:
            for status in StatusEffect.__members__.values():
                for i in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, status, HEALTH_MAX)
                    writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], attempt_catch(pokemon, pokeball, noise)[0], status._name_])

def analyze_2a_status():
        data = pd.read_csv(f"output/2a/status/{os.listdir('output/2a/status').pop()}")
        pokemon = data['Pokemon'].unique()
        #title = '2a. Efecto de la salud sobre la efectividad de captura ' + ', '.join(pokemon)
        data['Probability'] = pd.to_numeric(data['Probability'], errors='coerce')
        
        capture_mean = data.groupby(['Status', 'Pokeball'])['Probability'].mean().unstack()

        plt.figure(figsize=(12, 8))
        markers = ['o', '^', 's', 'p'] 
        line_styles = ['-', '--', '-.', ':']
        for idx, pokeball in enumerate(capture_mean.columns):
            plt.plot(capture_mean.index, capture_mean[pokeball], 
                    marker=markers[idx % len(markers)], 
                    linestyle=line_styles[idx % len(line_styles)], 
                    label=str(pokeball))
        #plt.title(title)
        plt.xlabel('Salud', fontsize=16)
        plt.ylabel('Probabilidad de captura', fontsize=16)
        plt.legend(title='Pokeball', loc='upper center', fontsize=14, title_fontsize=16)  
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(SAVE_PATH + EJ2A + '/2a_status.png')
        plt.close()

def analyze_2a_health():
    # Load data from CSV
    data = pd.read_csv(f"output/2a/health/2a.csv")
    pokemon = data['Pokemon'].unique()
    #title = '2a. Efecto de la salud sobre la efectividad de captura ' + ', '.join(pokemon)
    
    capture_mean = data.groupby(['Current HP', 'Pokeball'])['Captured'].mean().unstack()

    plt.figure(figsize=(12, 8))
    markers = ['o', '^', 's', 'p'] 
    line_styles = ['-', '--', '-.', ':']
    for idx, pokeball in enumerate(capture_mean.columns):
        x = capture_mean.index
        y = capture_mean[pokeball]
        
        # Interpolation for smooth lines
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
        y_smooth = spl(x_smooth)
        
        plt.plot(x_smooth, y_smooth, 
                 linestyle=line_styles[idx % len(line_styles)], 
                 label=str(pokeball))
        plt.scatter(x, y, 
                    marker=markers[idx % len(markers)], 
                    label=str(pokeball))

    #plt.title(title, fontsize=20)
    plt.xlabel('HP', fontsize=16)
    plt.ylabel('Probabilidad de captura', fontsize=16)
    plt.legend(title='Pokeball', loc='upper right', fontsize=14, title_fontsize=16)  
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(SAVE_PATH + EJ2A + '/2a_health.png')
    plt.close()


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    config_dir = sys.argv[1]
    pokemon_configs = os.listdir(config_dir)
    for pokemon_json in pokemon_configs:
        with open(f"{config_dir}/{pokemon_json}", "r") as f:
            config = json.load(f)
            pokemon = config["pokemon"]
            pokeballs = config["pokeballs"]
            load_2a_health(pokemon, pokeballs)
            load_2a_status(pokemon, pokeballs)

    analyze_2a_health()
    analyze_2a_status()
