import sys
from game.game import start_game
from utils.config_loader import ConfigLoader
from genetic_algorithm import algorithm
from game.eve import eve
from utils.time_manager import TimeManager

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    game_config_file = "../config/game_config.json"
    config_file = sys.argv[1]

    # Use ConfigLoader to load and validate the configuration
    config_loader = ConfigLoader(config_file, game_config_file)

    game_config = config_loader.load_game_config()
    timer, points, character = start_game(game_config)

    config = config_loader.load()
    algorithm.run_genetic_algorithm(config, eve, timer, points, character)


    sys.exit(1)

if __name__ == "__main__":
    main()
