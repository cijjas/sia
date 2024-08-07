import json
import sys
import csv
import matplotlib.pyplot as plt
import pandas as pd
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect



def load_catch_attempts(pokemon, pokeballs, noise=0):
    with open('output/catch_attempts.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Pokeball','Pokemon', 'Probability', 'Current HP'])
        for pokeball in pokeballs:
            for i in range(1000, 0, -1):  
                pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)
                writer.writerow([pokeball, pokemon.name,  attempt_catch(pokemon, pokeball, noise)[1], pokemon.current_hp])

def analyze_catch_attempts():
    data = pd.read_csv('output/catch_attempts.csv')
    mean_per_pokeball = data.groupby(['Pokeball'])['Probability'].agg(['mean', 'std'])
    
    plt.figure(figsize=(10, 5))
    
    plt.bar(mean_per_pokeball.index, mean_per_pokeball['mean'],  color='skyblue')
    plt.errorbar(mean_per_pokeball.index, mean_per_pokeball['mean'], yerr=mean_per_pokeball['std'], fmt='o', color='black', capsize=3)
    plt.title('Promedio de probabilidad de captura por tipo de pokeball')
    plt.xlabel('Pokeball')
    plt.ylabel('Probabilidad de captura')
    plt.grid(True)
    plt.xticks(rotation=0)

    plt.show()

def delete_file():
    import os
    if os.path.exists("output/catch_attempts.csv"):
        os.remove("output/catch_attempts.csv")

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        health_percentage = 1
        level = 100

        pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)

        # 1000 intentos de captura por cada pokeball
        load_catch_attempts(pokemon, config["pokeballs"], 0.15)
        analyze_catch_attempts()





        








