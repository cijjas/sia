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
SELECTION_RATE_FILE_CONF = "../config/algorithm_config.json"

def selection_method_comparison():
    """ Compare the selection methods using all available methods. Returns a csv with the best fitness, the average fitness and the execution time for each method."""
    config_file = "../config/selection_method_comparison.json"
    game_config_file = "../config/game_config.json"
    initial_population = "../config/placeholder.json"

    config_loader = ConfigLoader(config_file, game_config_file, initial_population_file=initial_population)
    game_config = config_loader.load_game_config()

    timer, points, character = start_game(game_config)

    config: Hyperparameters = config_loader.load()
    selection_methods_list = get_selection_methods()

    initial_population = config_loader.load_initial_population()
    selection_methods = []
    for method in selection_methods_list:
        selection_methods.append({"method": method})

     # create the output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_file = f"{OUTPUT_DIR}/selection_method_comparison.csv"

    # Open the file once and write the header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strength", "dexterity", "intelligence", "vigor", "constitution", "height", "fitness", "generation", "parent_selection_method", "replacement_selection_method"])

    for parent_selection_method in selection_methods:
        for replacement_selection_method in selection_methods:
            config.parents_selection_methods = [Selector(parent_selection_method)]
            config.replacements_selection_methods = [Selector(replacement_selection_method)]

            population = algorithm.run_genetic_algorithm(config, eve, timer, points, character, initial_population=initial_population)
            
            with open(output_file, "a", newline="") as f:  # Open the file in append mode
                writer = csv.writer(f)
                for individual in population.individuals:
                    writer.writerow([individual.genes.strength, individual.genes.dexterity, individual.genes.intelligence,
                                    individual.genes.vigor, individual.genes.constitution, individual.genes.height,
                                    individual.fitness, population.generation, 
                                    parent_selection_method["method"], replacement_selection_method["method"]])

    return

def selection_method_comparison_2():
    """ Compare the selection methods using all available methods. Returns a csv with the best fitness, the average fitness and the execution time for each method."""
    config_file = "../config/selection_method_comparison.json"
    game_config_file = "../config/game_config.json"

    config_loader = ConfigLoader(config_file, game_config_file)
    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config: Hyperparameters = config_loader.load()
    selection_methods_list = get_selection_methods()

    selection_methods = []
    for method1 in selection_methods_list:
        for method2 in selection_methods_list:
            selection_methods.append([
                {"method": method1, "weight": 0.5},
                {"method": method2, "weight": 0.5}
            ])

    # create the output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_file = f"{OUTPUT_DIR}/selection_method_comparison_2.csv"

    # Open the file once and write the header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strength", "dexterity", "intelligence", "vigor", "constitution", "height", "fitness", "generation",
                         "parent_selection_method_1", "parent_selection_method_2", "replacement_selection_method_1", "replacement_selection_method_2"])

    for parent_selection_methods in selection_methods:
        for replacement_selection_methods in selection_methods:
            config.parents_selection_methods = [Selector(parent_selection_methods[0]), Selector(parent_selection_methods[1])]
            config.replacements_selection_methods = [Selector(replacement_selection_methods[0]), Selector(replacement_selection_methods[1])]

            population = algorithm.run_genetic_algorithm(config, eve, timer, points, character)
            
            with open(output_file, "a", newline="") as f:  # Open the file in append mode
                writer = csv.writer(f)
                for individual in population.individuals:
                    writer.writerow([individual.genes.strength, individual.genes.dexterity, individual.genes.intelligence,
                                    individual.genes.vigor, individual.genes.constitution, individual.genes.height,
                                    individual.fitness, population.generation, 
                                    parent_selection_methods[0]["method"], parent_selection_methods[1]["method"],
                                    replacement_selection_methods[0]["method"], replacement_selection_methods[1]["method"]])

    return

def selection_rate_analysis(config_file):
    """ Compare the selection rate given a configuration. Returns a csv with the best fitness and the average fitness for each selection rate. """
    game_config_file = "../config/game_config.json"

    config_loader = ConfigLoader(config_file, game_config_file)
    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config: Hyperparameters = config_loader.load()

    # create the output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_file = f"{OUTPUT_DIR}/selection_rate_comparison.csv"

    # Open the file once and write the header
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strength", "dexterity", "intelligence", "vigor", "constitution", "height", "fitness", "generation", "selection_rate"])

    for selection_rate in range(1, 101):
        config.selection_rate = selection_rate / 100

        population = algorithm.run_genetic_algorithm(config, eve, timer, points, character)
        
        with open(output_file, "a", newline="") as f:  # Open the file in append mode
            writer = csv.writer(f)
            for individual in population.individuals:
                writer.writerow([individual.genes.strength, individual.genes.dexterity, individual.genes.intelligence,
                                individual.genes.vigor, individual.genes.constitution, individual.genes.height,
                                individual.fitness, population.generation, selection_rate])

    return

def main():
    if len(sys.argv) != 2:
        print("Using default files")
        args = ["", SELECTION_RATE_FILE_CONF]
    else:
        args = sys.argv
    selection_rate_analysis(args[1])
    selection_method_comparison()
    selection_method_comparison_2()
    print("Done")

if __name__ == "__main__":
    main()

