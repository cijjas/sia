import random
import time
from utils.time_manager import TimeManager

def start_game(game_config):

    if game_config["seed"]:
        time_limit = game_config["seed"]["time"]
        points = game_config["seed"]["points"]
        character = game_config["seed"]["character"]
    else:
        characters = game_config['character_classes']
        points_range = game_config['points_range']
        time_limit_range = game_config['time_limit_range']

        time_limit = random.randint(time_limit_range[0], time_limit_range[1]) # seconds
        points = random.randint(points_range[0], points_range[1])
        character = random.choice(characters)

    time_manager = TimeManager(time_limit)

    return time_manager, points, character






