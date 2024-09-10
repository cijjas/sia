import sys
import csv
import os
from game.game import start_game
from utils.config_loader import ConfigLoader
from genetic_algorithm import algorithm
from utils.eve import eve
from utils.time_manager import TimeManager
#from genetic_algorithm.selection import get_selection_methods
from genetic_algorithm.classes.hyperparameters import Hyperparameters, Selector

# OUTPUT_DIR = "../output"

# def selection_method_comparison():
#     """ Compare the selection methods using all available methods. Returns a csv with the best fitness, the average fitness and the execution time for each method."""
#     config_file = "../config/selection_method_comparison.json"
#     game_config_file = "../config/game_config.json"

#     config_loader = ConfigLoader(config_file, game_config_file)
#     game_config = config_loader.load_game_config()
#     timer, points, character = start_game(game_config)

#     config: Hyperparameters = config_loader.load()
#     selection_methods_list = get_selection_methods()

#     selection_methods = []
#     for method in selection_methods_list:
#         selection_methods.append({"method": method})

#     # create the output directory if it does not exist
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     output_file = f"{OUTPUT_DIR}/selection_method_comparison.csv"

    # Open the file once and write the header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strength", "dexterity", "intelligence", "vigor", "constitution", "height", "fitness", "generation", "parent_selection_method", "replacement_selection_method"])

#     for parent_selection_method in selection_methods:
#         for replacement_selection_method in selection_methods:
#             config.parents_selection_methods = [Selector(parent_selection_method)]
#             config.replacements_selection_methods = [Selector(replacement_selection_method)]

#             population = algorithm.run_genetic_algorithm(config, eve, timer, points, character, show=False)
            
            with open(output_file, "a", newline="") as f:  # Open the file in append mode
                writer = csv.writer(f)
                for individual in population.individuals:
                    writer.writerow([individual.genes.strength, individual.genes.dexterity, individual.genes.intelligence,
                                     individual.genes.vigor, individual.genes.constitution, individual.genes.height,
                                     individual.fitness, population.generation, 
                                     parent_selection_method["method"], replacement_selection_method["method"]])

#     return


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    game_config_file = "../config/game_config.json" # hardcodeado porque no tiene sentido no
    config_file = sys.argv[1]

    # Use ConfigLoader to load and validate the configuration
    config_loader = ConfigLoader(config_file, game_config_file)

    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config = config_loader.load()
    
    best_option = algorithm.run_genetic_algorithm(config, eve, timer, points, character)

    print("Your best option is: ", best_option)

    sys.exit(1)

if __name__ == "__main__":
    # selection_method_comparison()
    main()
