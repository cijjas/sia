import sys
import csv
import os
from game.game import start_game
from utils.config_loader import ConfigLoader
from genetic_algorithm import algorithm
from utils.eve import eve
from genetic_algorithm.selection import get_selection_methods
from genetic_algorithm.classes.hyperparameters import Hyperparameters, Selector

OUTPUT_DIR = "../output"
ALGORITHM_CONF = "../config/algorithm_config.json"

def mutation_rate_analysis(config_file):
    """ Compare the mutation rate given a configuration. Returns a csv with the best fitness and the average fitness for each mutation rate. """
    game_config_file = "../config/game_config.json"
    mutation_rates_list = [0, 0.5, 0.95]

    config_loader = ConfigLoader(config_file, game_config_file)
    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config: Hyperparameters = config_loader.load()

    # create the output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_file = f"{OUTPUT_DIR}/mutation_rate_comparison.csv"

    # Open the file once and write the header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strength", "dexterity", "intelligence", "vigor", "constitution", "height", "fitness", "generation", "mutation_rate"])
        
    for mutation_rate in mutation_rates_list:
        config.mutation_rate = mutation_rate

        population = algorithm.run_genetic_algorithm(config, eve, timer, points, character)
        
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            for individual in population.individuals:
                writer.writerow([individual.genes.strength, individual.genes.dexterity, individual.genes.intelligence,
                                individual.genes.vigor, individual.genes.constitution, individual.genes.height,
                                individual.fitness, population.generation, mutation_rate])
                
    return

def main():
    if len(sys.argv) != 2:
        print("Using default files")
        args = ["", ALGORITHM_CONF]
    else:
        args = sys.argv
    mutation_rate_analysis(args[1])
    print("Done")

if __name__ == "__main__":
    main()

