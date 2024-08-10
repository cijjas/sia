import json
import sys
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from src.catching import attempt_catch
from matplotlib.colors import LinearSegmentedColormap  # Add this import
from scipy.interpolate import make_interp_spline

from src.pokemon import PokemonFactory, StatusEffect

LEVEL_MAX = 100
HEALTH_MAX = 1
SAVE_PATH = 'output/analyze/'
EJ1A = '1a'
EJ1B = '1b'
EJ2A = '2a'
EJ2B = '2b'
EJ2C = '2c'


def load_1(pokemonName, pokeballs, directory, noise=0, reps=100):

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

    # Set global font size
    plt.rcParams.update({'font.size': 20})

    # agrupo por pokeball y pokemon y saco la media de catch success
    grouped_data = all_data.groupby(['Pokeball', 'Pokemon'])['Catch Success'].mean().unstack()

    # Probabilidad promedio de captura por tipo de pokeball para cada pokemon
    ax = grouped_data.plot(kind='bar', figsize=(14, 8))
    ax.set_xlabel('Pokeball')
    ax.set_ylabel('Probabilidad de captura')
    ax.grid(True, linestyle='--')
    ax.legend(title='Pokemon')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SAVE_PATH + EJ1A + '/promedio_captura_por_pokebola_por_pokemon.png')
    plt.close()

    # Promedio de captura por tipo de pokeball para todos los pokemones con error estandar 
    long_data = grouped_data.reset_index().melt(id_vars='Pokeball', var_name='Pokemon', value_name='Probability')

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Pokeball', y='Probability', data=long_data, palette="Set3", hue='Pokeball')
    plt.xlabel('Pokéball')
    plt.ylabel('Probabilidad de captura')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(SAVE_PATH+ EJ1A + '/promedio_captura_con_error.png')
    plt.close()

    # boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Pokeball', y='Probability', data=long_data, palette="Set3")
    plt.xlabel('Pokéball')
    plt.ylabel('Probabilidad de captura')
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(SAVE_PATH + EJ1A+ '/promedio_captura_boxplot.png')
    plt.close()


    # Varianza de las probabilidades de captura para cada pokeball
    variance_data = all_data.groupby('Pokeball')['Catch Success'].var()
    plt.figure(figsize=(10, 6))
    variance_data.plot(kind='line', color='tomato', marker='o')
    plt.xlabel('Pokeball')
    plt.ylabel('Varianza')
    plt.grid(True, linestyle='--')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SAVE_PATH + EJ1A+ '/varianza.png')
    plt.close()

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
    # print(catch_success_for_pokeball)

    # Probabilidad promedio de captura por tipo de pokeball para cada pokemon
    ax = grouped_data.plot(kind='bar', figsize=(14, 8))
    ax.set_xlabel('Pokeball')
    ax.set_ylabel('Probabilidad de captura')
    ax.grid(True, linestyle='--')
    ax.legend(title='Pokemon')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SAVE_PATH + EJ1B+ '/1b.png')
    plt.close()


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
    plt.savefig(SAVE_PATH + EJ2B+ '/2b.png')
    plt.close()


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

def analyze_2c():
    all_data = pd.DataFrame()

    # Cargar todos los archivos CSV del directorio output/2c
    for filename in os.listdir('output/2c'):
        data = pd.read_csv(f'output/2c/{filename}')
        all_data = pd.concat([all_data, data], ignore_index=True)

    custom_cmap = LinearSegmentedColormap.from_list("color scale", ["purple", "blue", "green", "yellow"])

    # Generar un heatmap para cada combinación de Status Effect y Pokeball
    for status in all_data['Status Effect'].unique():
        for pokeball in all_data['Pokeball'].unique():
            subset = all_data[(all_data['Status Effect'] == status) & (all_data['Pokeball'] == pokeball)]
            if not subset.empty:
                heatmap_data = subset.pivot_table(values='Catch Success', index='Level', columns='%HP', aggfunc='mean')

                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, fmt=".2f", cmap=custom_cmap, vmin=0, vmax=1)
                plt.title(f'Heatmap: Average Catch Success\nPokemon: {subset.iloc[0]["Pokemon"]}, Status Effect: {status}, Pokeball: {pokeball}')
                plt.xlabel('%HP')
                plt.ylabel('Level')
                plt.tight_layout()
                plt.savefig(SAVE_PATH + EJ2C+ f'/{pokeball}_{status}.png')
                plt.close()

def load_2c(pokemonName, pokeballs, noise=0):
    with open(f'output/2c/{pokemonName}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball', 'Pokemon', 'Level', '%HP', 'Status Effect', 'Catch Success'])

        for level in range(100, 0, -1):
            for status in StatusEffect.__members__.values():
                for pokeball in pokeballs:
                    for hp_percent in range(100, 0, -1):
                        pokemon = factory.create(pokemonName, level, status, hp_percent/100)
                        catch_success = attempt_catch(pokemon, pokeball, noise)[1]
                        writer.writerow([pokeball, pokemon.name, level, hp_percent/100 , status, catch_success])


def load_2c_bis(pokemonName, pokeballs, noise=0, reps=100):
    parameters = ["HP", "Level", "Status Effect", "Pokeball"]
    
    for parameter in parameters:
        with open(f'output/2c/comp/{pokemonName}-{parameter}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pokeball', 'Pokemon', 'Level', '%HP', 'Status Effect', 'Catch Success'])
            
            default_level = LEVEL_MAX
            default_hp_percent = HEALTH_MAX
            default_status = StatusEffect.NONE
            default_pokeball = "pokeball"

            if parameter == "HP":
                for hp_percent in range(reps, 0, -1):
                    pokemon = factory.create(pokemonName, default_level, default_status, hp_percent / 100)
                    catch_success = attempt_catch(pokemon, default_pokeball, noise)[1]
                    writer.writerow([default_pokeball, pokemon.name, default_level, hp_percent / 100, default_status, catch_success])
            
            elif parameter == "Level":
                for level in range(reps, 0, -1):
                    pokemon = factory.create(pokemonName, level, default_status, default_hp_percent)
                    catch_success = attempt_catch(pokemon, default_pokeball, noise)[1]
                    writer.writerow([default_level, pokemon.name, level, default_hp_percent, default_status, catch_success])
            
            elif parameter == "Status Effect":
                for hp_percent in range(reps, 0, -1):
                    for status in StatusEffect.__members__.values():
                        pokemon = factory.create(pokemonName, default_level, status, default_hp_percent)
                        catch_success = attempt_catch(pokemon, default_pokeball, noise)[1]
                        writer.writerow([default_pokeball, pokemon.name, default_level, default_hp_percent, status._name_, catch_success])
            
            elif parameter == "Pokeball":
                for hp_percent in range(reps, 0, -1):
                    for pokeball in pokeballs:
                        pokemon = factory.create(pokemonName, default_level, default_status, default_hp_percent)
                        catch_success = attempt_catch(pokemon, pokeball, noise)[1]
                        writer.writerow([pokeball, pokemon.name, default_level, default_hp_percent, default_status, catch_success])

def analyze_2c_bis():
    # List all files in the output/2c/ directory
    files = os.listdir('output/2c/comp/')
    for file in files:
        # Extract pokemon name and parameter from the filename
        filename_parts = file.split('-')
        pokemon = filename_parts[0]
        parameter = filename_parts[1].replace('.csv', '')

        # Load the CSV file
        file_path = f'output/2c/comp/{file}'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            # Generate plots based on the parameter
            if parameter == "HP":
                # agrupo por pokeball y pokemon y saco la media de catch success
                (hp, probability) =(data['%HP'] ,data['Catch Success']) 
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=hp, y=probability, data=data)
                plt.title(f'2c1: {pokemon}: HP vs Catch Success')
                plt.xlabel('%HP')
                plt.ylabel('Catch Success')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(SAVE_PATH+ EJ2C + f'/{pokemon}_{parameter}.png')
                plt.close()

            elif parameter == "Level":
                (level, probability) = (data['Level'], data['Catch Success'])
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=level, y=probability, data=data)
                plt.title(f'2c2: {pokemon}: Level vs Catch Success')
                plt.xlabel('Level')
                plt.ylabel('Catch Success')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(SAVE_PATH + EJ2C+ f'/{pokemon}_{parameter}.png')
                plt.close()

            elif parameter == "Pokeball":
                # agrupo por pokeball y pokemon y saco la media de catch success
                (pokeball, probability) = (data['Pokeball'], data['Catch Success'])
                plt.figure(figsize=(10, 6))
                sns.barplot(x=pokeball, y=probability, data=data)
                plt.title(f'2c3: {pokemon}: Pokeball vs Catch Success')
                plt.xlabel('Pokeball')
                plt.ylabel('Catch Success')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(SAVE_PATH + EJ2C+ f'/{pokemon}_{parameter}.png')
                plt.close()

            elif parameter == "Status Effect":
                # agrupo por pokeball y pokemon y saco la media de catch success
                (status, probability) = (data['Status Effect'], data['Catch Success'])
                plt.figure(figsize=(10, 6))
                sns.barplot(x=status, y=probability, data=data)
                plt.title(f'2c4: {pokemon}: Status Effect vs Catch Success')
                plt.xlabel('Status Effect')
                plt.ylabel('Catch Success')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(SAVE_PATH + EJ2C+ f'/{pokemon}_{parameter}.png')
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
            load_1(pokemon, pokeballs, directory='output/1a')
            load_1(pokemon, pokeballs, directory='output/1b', reps=100000)
            load_2a_health(pokemon, pokeballs)
            load_2a_status(pokemon, pokeballs)
            load_2b(pokemon, pokeballs)
            load_2c(pokemon, pokeballs)
            load_2c_bis(pokemon, pokeballs)
    
    analyze_1a()
    analyze_1b()
    analyze_2a_health()
    analyze_2a_status()
    analyze_2b()
    analyze_2c_bis()
    analyze_2c()
