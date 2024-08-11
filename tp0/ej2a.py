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
        tries = 500
        for pokeball in pokeballs:
            for i in range(divisions, 0, -1):
                for j in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/divisions)
                    attempt = attempt_catch(pokemon, pokeball, noise)
                    writer.writerow([pokeball, pokemon.name,  attempt[1], attempt[0], i/divisions])
    # now we generate the data for the noise analysis
    with open('output/2a/health/2a_noise_1.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Captured', 'Current HP'])
        divisions = 20
        tries = 500
        for pokeball in pokeballs:
            for i in range(divisions, 0, -1):
                for j in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/divisions)
                    attempt = attempt_catch(pokemon, pokeball, 0.1)
                    writer.writerow([pokeball, pokemon.name,  attempt[1], attempt[0], i/divisions])
    with open('output/2a/health/2a_noise_2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Captured', 'Current HP'])
        divisions = 20
        tries = 500
        for pokeball in pokeballs:
            for i in range(divisions, 0, -1):
                for j in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, StatusEffect.NONE, i/divisions)
                    attempt = attempt_catch(pokemon, pokeball, 0.1)
                    writer.writerow([pokeball, pokemon.name,  attempt[1], attempt[0], i/divisions])

def load_2a_status(pokemonName, pokeballs, noise=0):
    with open(f"output/2a/status/{pokemonName}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Captured', 'Status'])
        tries = 1000
        for pokeball in pokeballs:
            for status in StatusEffect.__members__.values():
                for i in range(tries):
                    pokemon = factory.create( pokemonName, LEVEL_MAX, status, HEALTH_MAX)
                    attempt = attempt_catch(pokemon, pokeball, noise)
                    writer.writerow([pokeball, pokemon.name,  attempt[1], attempt[0], status._name_])

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
    data_noise_1 = pd.read_csv(f"output/2a/health/2a_noise_1.csv")
    data_noise_2 = pd.read_csv(f"output/2a/health/2a_noise_2.csv")
    pokemon = data['Pokemon'].unique()
    #title = '2a. Efecto de la salud sobre la efectividad de captura ' + ', '.join(pokemon)
    
    capture_mean = data.groupby(['Current HP', 'Pokeball'])['Captured'].mean().unstack()
    capture_mean_noise_1 = data_noise_1.groupby(['Current HP', 'Pokeball'])['Captured'].mean().unstack()
    capture_mean_noise_2 = data_noise_2.groupby(['Current HP', 'Pokeball'])['Captured'].mean().unstack()

    plt.figure(figsize=(12, 8))
    markers = ['o', '^', 's', 'p'] 
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple']  # Add more colors if needed
    
    for idx, pokeball in enumerate(capture_mean.columns):
        x = capture_mean.index
        y = capture_mean[pokeball]
        y_noise_1 = capture_mean_noise_1[pokeball]
        y_noise_2 = capture_mean_noise_2[pokeball]
        
        # Calculate upper and lower bounds for the shaded area for both noise datasets
        lower_bound_1 = y - np.abs(y - y_noise_1)
        upper_bound_1 = y + np.abs(y - y_noise_1)
        lower_bound_2 = y - np.abs(y - y_noise_2)
        upper_bound_2 = y + np.abs(y - y_noise_2)
        
        # Calculate the overall lower and upper bounds
        lower_bound = np.minimum(lower_bound_1, lower_bound_2)
        upper_bound = np.maximum(upper_bound_1, upper_bound_2)
        
        # Select color for the current pokeball
        color = colors[idx % len(colors)]
        
        # Plot the mean line
        plt.plot(x, y, 
                 marker=markers[idx % len(markers)], 
                 linestyle=line_styles[idx % len(line_styles)], 
                 color=color,
                 label=str(pokeball))
        
        # Fill the area between the overall bounds with the same color
        plt.fill_between(x, lower_bound, upper_bound, alpha=0.2, color=color)
    
    #plt.title(title)
    plt.xlabel('Salud', fontsize=16)
    plt.ylabel('Probabilidad de captura', fontsize=16)
    plt.legend(title='Pokeball', loc='upper center', fontsize=14, title_fontsize=16)
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
            load_2a_health(pokemon, pokeballs, noise=0.2)
            #load_2a_status(pokemon, pokeballs)

    analyze_2a_health()
    #analyze_2a_status()
