import sys
from game.game import start_game
from utils.config_loader import ConfigLoader
from genetic_algorithm import algorithm
from utils.eve import eve


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    game_config_file = "../config/game_config.json" # hardcodeado porque no tiene sentido no
    config_file = sys.argv[1]
    initial_population_config_file = "../config/placeholder.json"

    # Use ConfigLoader to load and validate the configuration
    config_loader = ConfigLoader(config_file, game_config_file, initial_population_file=initial_population_config_file)

    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config = config_loader.load()
    initial_population_config = config_loader.load_initial_population()
    
    population = algorithm.run_genetic_algorithm(config, eve, timer, points, character, 
                                                 initial_population=initial_population_config, debug=True)

    print("Your best option is: ", population.best_individual)


if __name__ == "__main__":
    main()
