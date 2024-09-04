import sys
from game.game import start_game
from tp2.src.utils.config_loader import ConfigLoader
from game.eve import EVE
import genetic_algorithm.algorithm as algorithm

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    game_config_file = "../config/game_config.json"
    config_file = sys.argv[1]

    try:
        # Use ConfigLoader to load and validate the configuration
        config_loader = ConfigLoader(config_file, game_config_file)
        config = config_loader.load()
        game_config = config_loader.load_game_config() 

        timer = start_game(config, game_config)

        algorithm.run_genetic_algorithm(config, EVE(), timer)


    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
