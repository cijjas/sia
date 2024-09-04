import sys
from game import start
from config_loader import ConfigLoader

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    try:
        # Use ConfigLoader to load and validate the configuration
        config_loader = ConfigLoader(config_file)
        config = config_loader.load()

        # Start the game with the loaded configuration
        start(config)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
